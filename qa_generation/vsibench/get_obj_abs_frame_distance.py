"""
The format of the qa file is as follows:
{
    "id": id,
    "scene_name": "xxxx",
    "question_type": "the question type a.k.a. the task at hand",
    "video_path": "path to the video",
    "category, first object": "category of object 1",
    "category, second object": "category of object 2",
    "question": "question",
    "options": "answer options",
    "ground_truth": "either the float or string answer",
    "mc_answer": "letter answer",
}
"""

from preprocessing.preprocess_scannet.common_utils import (
    cal_3d_bbox_distance_between_categories,
    generate_multiple_choice,
)
from ..base_qa_generator import BaseQAGenerator


class ObjAbsDistanceQAGenerator(BaseQAGenerator):
    def get_default_args(self):
        return {
            'question_template': "OBJ_ABS_DISTANCE_FRAME_TEMPLATE",   # changed
            'num_subsample': 6,
            'question_type': 'object_abs_distance_frame',             # changed (distinct type)
            'output_filename_prefix': 'qa_obj_abs_distance_frame'     # changed (distinct file)
        }

    def _get_all_instance_details_map(self, scene_name, scene_info):
        """
        Build a map: instance_id -> {'category_name': str, 'bbox_meta': dict}
        Requires scene_info['object_bboxes'][category] to be a list of dicts
        that include keys: instance_id, centroid, normalizedAxes, axesLengths.
        """
        all_instance_details = {}
        object_bboxes_dict = scene_info.get('object_bboxes', {})

        if not object_bboxes_dict:
            return {}

        required_keys = ["centroid", "normalizedAxes", "axesLengths"]

        for category, instances in object_bboxes_dict.items():
            if not isinstance(instances, list):
                continue
            for obj in instances:
                if not isinstance(obj, dict):
                    continue
                iid = obj.get("instance_id")
                if iid is None:
                    continue
                if not all(k in obj for k in required_keys):
                    continue
                all_instance_details[int(iid)] = {
                    "category_name": category,
                    "bbox_meta": {k: obj[k] for k in required_keys}
                }
        return all_instance_details

    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """
        Generate frame-wise absolute distance QA pairs.
        For each frame, consider categories that are visible exactly once in that frame,
        then create pairwise questions between those frame-unique categories.
        """
        scene_qa_list = []

        video_path = scene_info.get('video_path')
        frames = frame_info_for_scene.get('frames', [])
        if not frames:
            return scene_qa_list

        total_frames_in_meta = len(frames)

        # Build a quick lookup: instance_id -> category_name
        iid_to_category = {}
        obj_bboxes = scene_info.get('object_bboxes', {})
        for category, instances in obj_bboxes.items():
            if not isinstance(instances, list):
                continue
            for obj in instances:
                if not isinstance(obj, dict):
                    continue
                iid = obj.get('instance_id')
                if iid is None:
                    continue
                iid_to_category[int(iid)] = category

        if not iid_to_category:
            return scene_qa_list

        # Iterate frames in order of frame_id
        frames_sorted = sorted(frames, key=lambda f: f['frame_id'])

        for rank, f in enumerate(frames_sorted):
            frame_id = f.get('frame_id')
            b2d = f.get('bboxes_2d', [])
            if not b2d:
                continue

            # Count categories visible exactly once in THIS frame (by instance id → category)
            cat_counts = {}
            cat_single_iid = {}

            for box in b2d:
                iid = box.get('instance_id')
                if iid is None:
                    continue
                iid = int(iid)
                category = iid_to_category.get(iid)
                if category is None:
                    continue
                cat_counts[category] = cat_counts.get(category, 0) + 1
                # Keep track of the instance id for singletons (we'll use it only if count==1)
                if category not in cat_single_iid:
                    cat_single_iid[category] = iid

            # Keep only categories with exactly ONE visible instance this frame
            unique_cats = [c for c, cnt in cat_counts.items() if cnt == 1]
            if len(unique_cats) < 2:
                continue

            # Create all unique pairs (A,B) with A < B in index to avoid duplicates
            for i in range(len(unique_cats)):
                for j in range(i + 1, len(unique_cats)):
                    cat_a = unique_cats[i]
                    cat_b = unique_cats[j]

                    iid_a = cat_single_iid[cat_a]
                    iid_b = cat_single_iid[cat_b]

                    # Pull only the exact instance dicts to feed the distance util
                    objs_a_all = obj_bboxes.get(cat_a, [])
                    objs_b_all = obj_bboxes.get(cat_b, [])

                    inst_a_list = [o for o in objs_a_all if int(o.get('instance_id', -1)) == int(iid_a)]
                    inst_b_list = [o for o in objs_b_all if int(o.get('instance_id', -1)) == int(iid_b)]
                    if not inst_a_list or not inst_b_list:
                        continue

                    # Compute min distance between these two *specific* instances
                    dist = cal_3d_bbox_distance_between_categories(inst_a_list, inst_b_list)
                    if dist is None:
                        continue

                    ground_truth = round(float(dist), 1)
                    if ground_truth < 0.2:  # too small / unreliable
                        continue

                    # Build the question text with frame context — safe even if template ignores it
                    frame_desc = f"frame {rank + 1} of {total_frames_in_meta}"
                    question_text = self.question_template.format(
                        object1=cat_a,
                        object2=cat_b,
                        frame_description=frame_desc
                    )

                    # Generate balanced multiple-choice options and letter
                    options_raw, mc_answer, self.answer_counts = generate_multiple_choice(
                        ground_truth,
                        decimals=1,
                        answer_counts=self.answer_counts,
                        option_letters=self.option_letters
                    )
                    if mc_answer == "E":
                        continue

                    # Match original "A. ..." formatting
                    options_printable = [
                        f"A. {options_raw[0]}",
                        f"B. {options_raw[1]}",
                        f"C. {options_raw[2]}",
                        f"D. {options_raw[3]}",
                    ]

                    qa = {
                        'dataset': self.args.dataset,
                        "scene_name": scene_name,
                        "video_path": video_path,
                        "frame_indices": [frame_id],                # frame-wise
                        "question_type": self.args.question_type,
                        "category, first object": cat_a,
                        "category, second object": cat_b,
                        "question": question_text,
                        "options": options_printable,
                        "ground_truth": ground_truth,
                        "mc_answer": mc_answer
                    }
                    scene_qa_list.append(qa)

        return scene_qa_list



if __name__ == '__main__':
    # Removed argparse logic, handled by BaseQAGenerator
    generator = ObjAbsDistanceQAGenerator()
    generator.run()