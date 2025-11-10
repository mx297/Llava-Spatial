"""
The format of the qa file is as follows:
{
    "id": id,
    "scene_name": "xxxx",
    "question_type": "the question type a.k.a. the task at hand",
    "video_path": "path to the video",
    "category: "category of the object",
    "question": "question",
    "options": "answer options",
    "ground truth": "either the float or string answer",
    "mc_answer": "letter answer",
}
"""
from preprocessing.preprocess_scannet.common_utils import generate_multiple_choice
from ..base_qa_generator import BaseQAGenerator


class ObjSizeQAGenerator(BaseQAGenerator):
    def get_default_args(self):
        return {
            'question_template': "OBJ_SIZE_ESTIMATE_FRAME_TEMPLATE",  # changed
            'num_subsample': 6,
            'question_type': 'object_size_estimation_frame',          # changed (distinct type)
            'output_filename_prefix': 'qa_obj_size_frame'             # changed (distinct file)
        }

    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """
        Generate frame-wise object size QA pairs.

        For each frame, identify categories that are visible exactly once in that frame
        (using bboxes_2d -> instance_id -> category). For each such category, use the
        specific instance's 3D bbox (axesLengths) from scene-level metadata to compute
        the object's longest dimension in centimeters and build a question bound to
        that frame.
        """
        scene_qa_list = []
        video_path = scene_info.get('video_path')

        frames = frame_info_for_scene.get('frames', [])
        if not frames:
            return scene_qa_list

        total_frames_in_meta = len(frames)

        # Build instance_id -> category mapping from scene-level metadata
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

        # Also build category -> {iid -> obj_dict} to quickly fetch the exact instance meta
        cat_iid_to_obj = {}
        for category, instances in obj_bboxes.items():
            if not isinstance(instances, list):
                continue
            cat_iid_to_obj[category] = {}
            for obj in instances:
                if not isinstance(obj, dict):
                    continue
                iid = obj.get('instance_id')
                if iid is None:
                    continue
                cat_iid_to_obj[category][int(iid)] = obj

        # Iterate frames in order
        frames_sorted = sorted(frames, key=lambda f: f['frame_id'])

        for rank, f in enumerate(frames_sorted):
            frame_id = f.get('frame_id')
            b2d = f.get('bboxes_2d', [])
            if not b2d:
                continue

            # Count visible instances per category in THIS frame;
            # use a set to avoid duplicate boxes for the same instance
            cat_to_iids = {}

            for box in b2d:
                iid = box.get('instance_id')
                if iid is None:
                    continue
                iid = int(iid)
                category = iid_to_category.get(iid)
                if category is None:
                    continue
                if category not in cat_to_iids:
                    cat_to_iids[category] = set()
                cat_to_iids[category].add(iid)

            # Keep only categories with exactly one visible instance this frame
            unique_cats = [c for c, s in cat_to_iids.items() if len(s) == 1]
            if not unique_cats:
                continue

            for category in unique_cats:
                iid = next(iter(cat_to_iids[category]))  # the single visible instance id
                inst_meta = cat_iid_to_obj.get(category, {}).get(iid)
                if not inst_meta:
                    continue

                # Read the OBB axis lengths (meters) and get the longest dimension
                scale = inst_meta.get('axesLengths')
                if not scale or not isinstance(scale, (list, tuple)):
                    continue
                try:
                    length_m = float(max(scale))
                except Exception:
                    continue

                length_cm = round(length_m * 100)  # convert to centimeters, integer

                # Generate balanced multiple-choice options (integer cm)
                options, mc_answer, self.answer_counts = generate_multiple_choice(
                    length_cm,
                    lower_bound=0.4,  # keep your original bounds
                    upper_bound=1.8,
                    decimals=0,       # integer centimeters
                    answer_counts=self.answer_counts,
                    option_letters=self.option_letters
                )
                if mc_answer == "E":
                    continue

                # Frame-aware text (safe even if template ignores frame_description)
                frame_desc = f"frame {rank + 1} of {total_frames_in_meta}"
                question_text = self.question_template.format(
                    category=category,
                    frame_description=frame_desc
                )

                qa = {
                    # "id" will be assigned by the base class run method
                    "scene_name": scene_name,
                    'dataset': self.args.dataset,
                    'question_type': self.args.question_type,
                    "video_path": video_path,
                    "frame_indices": [frame_id],  # frame-wise
                    'category': category,
                    "question": question_text,
                    "options": [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"],
                    "ground_truth": length_cm,
                    "mc_answer": mc_answer
                }
                scene_qa_list.append(qa)

        return scene_qa_list



if __name__ == '__main__':
    generator = ObjSizeQAGenerator()
    generator.run()