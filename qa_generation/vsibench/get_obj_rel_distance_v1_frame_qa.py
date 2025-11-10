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

import random
import itertools

from ..base_qa_generator import BaseQAGenerator
from preprocessing.preprocess_scannet.common_utils import cal_3d_bbox_distance_between_categories, from_options_to_mc_answer


def ambiguous(distances, threshold=0.15):
    """
    Check if any of the distance answers are within 15cm of each other, meaning that the ground truth answer would likely be ambiguous to a human annotator.

    Args:
        distances: list[float] of length 4 of the distances of each option to the queried object
        threshold: float, the distance threshold to check against (default is 0.15 meters).

    Returns:
        bool: True if any pair of objects are within the threshold distance, False otherwise.
    """
    
    # Compare each pair of choices
    for i in range(4):
        for j in range(i + 1, 4):
            if abs(distances[i] - distances[j]) <= threshold:
                return True
    
    return False


def get_gt(choice_categories, primary_object_category, obj_bbox_info, 
           dist_lookup, room_size, answer_counts, option_letters):
    '''
    Arg(s):
        choice_categories: list[str] of 4 chosen object choice categories
        primary_object_category: category of the primary object (to which we are trying to choose the closest object)
        scene_name: name of the scene

    Output:
        ground_truth: the letter (A, B, C, D) of the correct answer
    '''

    # distances = [cal_point_set_distance_between_categories(category, primary_object_category, scene_name) for category in choice_categories]
    # distances = [cal_3d_bbox_distance_between_categories(obj_bbox_info[category], obj_bbox_info[primary_object_category]) for category in choice_categories]
    distances = []
    for category in choice_categories:
        if (category, primary_object_category) in dist_lookup:
            dist = dist_lookup[(category, primary_object_category)]
        elif (primary_object_category, category) in dist_lookup:
            dist = dist_lookup[(primary_object_category, category)]
        else:
            dist = cal_3d_bbox_distance_between_categories(
                obj_bbox_info[category], obj_bbox_info[primary_object_category]
            )
            dist_lookup[(category, primary_object_category)] = dist
            dist_lookup[(primary_object_category, category)] = dist
            
        distances.append(dist)
        
    min_distance = min(distances)

    distance_to_object_threshold = 0.15
    if min_distance < distance_to_object_threshold:
        return "ambiguous", None, None, answer_counts

    min_index = distances.index(min_distance)

    # check if any of the point cloud instance lists were None
    if min_distance < 0:
        return "point cloud error", None, None, answer_counts
    
    threshold = 0.30 if room_size > 40 else 0.15
    if ambiguous(distances):
        return "ambiguous", None, None, answer_counts
    
    ground_truth = choice_categories[min_index]
    
    # Shuffle the choices
    options, mc_answer, answer_counts = from_options_to_mc_answer(
        choice_categories, ground_truth, answer_counts, option_letters
    )

    return ground_truth, mc_answer, options, answer_counts


class ObjRelDistQAGenerator(BaseQAGenerator):
    def get_default_args(self):
        return {
            'question_template': "OBJ_REL_DISTANCE_V1_FRAME_TEMPLATE",   # changed
            'num_subsample': 6,
            'question_type': 'object_rel_distance_v1_frame',              # changed
            'output_filename_prefix': 'qa_obj_rel_distance_v1_frame'      # changed
        }


    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """
        Generate frame-wise relative distance V1 QA pairs.

        For each frame, consider all visible object categories (from bboxes_2d) that appear exactly once.
        For each single-count category, select 4 other visible objects and ask:
        "Which of these is closest to the <category>?".
        """
        scene_qa_list = []
        video_path = scene_info.get('video_path')
        frames = frame_info_for_scene.get('frames', [])
        if not frames:
            return scene_qa_list

        total_frames_in_meta = len(frames)

        # Build instance_id -> category mapping from scene metadata
        iid_to_category = {}
        obj_bboxes = scene_info.get('object_bboxes', {})
        for category, instances in obj_bboxes.items():
            if not isinstance(instances, list):
                continue
            for obj in instances:
                iid = obj.get('instance_id')
                if iid is None:
                    continue
                iid_to_category[int(iid)] = category

        if not iid_to_category:
            return scene_qa_list

        frames_sorted = sorted(frames, key=lambda f: f['frame_id'])
        dist_lookup = {}

        for rank, f in enumerate(frames_sorted):
            frame_id = f.get('frame_id')
            b2d = f.get('bboxes_2d', [])
            if not b2d:
                continue

            # Count visible instances per category for this frame
            cat_to_iids = {}
            for box in b2d:
                iid = box.get('instance_id')
                if iid is None:
                    continue
                iid = int(iid)
                category = iid_to_category.get(iid)
                if category is None:
                    continue
                cat_to_iids.setdefault(category, set()).add(iid)

            # Keep only categories with exactly one visible instance this frame
            visible_cats = list(cat_to_iids.keys())
            single_visible_cats = [c for c, s in cat_to_iids.items() if len(s) == 1]

            if len(visible_cats) < 5 or not single_visible_cats:
                continue

            frame_qa_candidates = []
            for primary_cat in single_visible_cats:
                other_cats = list(set(visible_cats) - {primary_cat})
                if len(other_cats) < 4:
                    continue

                combos = list(itertools.combinations(other_cats, 4))
                for combo in combos:
                    frame_qa_candidates.append({
                        "primary_category": primary_cat,
                        "choice_categories": list(combo)
                    })

            if len(frame_qa_candidates) > self.args.num_subsample:
                frame_qa_candidates = random.sample(frame_qa_candidates, self.args.num_subsample)

            # Compute ground truth for each candidate
            for candidate in frame_qa_candidates:
                primary_cat = candidate["primary_category"]
                choice_cats = candidate["choice_categories"]
                room_size = scene_info.get('room_size', 30.0)  # fallback

                gt, mc_answer, options, self.answer_counts = get_gt(
                    choice_cats, primary_cat, obj_bboxes, dist_lookup,
                    room_size, self.answer_counts, self.option_letters
                )

                if gt in ("ambiguous", "point cloud error"):
                    continue

                frame_desc = f"frame {rank + 1} of {total_frames_in_meta}"
                question_text = self.question_template.format(
                    choice_a=options[0],
                    choice_b=options[1],
                    choice_c=options[2],
                    choice_d=options[3],
                    category=primary_cat,
                    frame_description=frame_desc
                )

                qa = {
                    # "id" will be assigned by the base class run method
                    "scene_name": scene_name,
                    'dataset': self.args.dataset,
                    "question_type": self.args.question_type,
                    "video_path": video_path,
                    "frame_indices": [frame_id],  # frame-wise link
                    "category": primary_cat,
                    "question": question_text,
                    "options": [
                        f"A. {options[0]}",
                        f"B. {options[1]}",
                        f"C. {options[2]}",
                        f"D. {options[3]}",
                    ],
                    "ground_truth": gt,
                    "mc_answer": mc_answer
                }
                scene_qa_list.append(qa)

        return scene_qa_list


if __name__ == '__main__':
    generator = ObjRelDistQAGenerator()
    generator.run()