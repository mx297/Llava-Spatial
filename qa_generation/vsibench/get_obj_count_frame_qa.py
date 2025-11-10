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
    "ground_truth": "either the float or string answer",
    "mc_answer": "letter answer",
}
"""

import random

from ..base_qa_generator import BaseQAGenerator
from preprocessing.preprocess_scannet.common_utils import from_options_to_mc_answer


def generate_choices(answer, answer_counts, option_letters):
    choices = [answer]
    
    # Create a set of possible offsets, excluding 0
    possible_offsets = set(range(-3, 4)) - {0}
    
    while len(choices) < 4:
        if not possible_offsets:
            # If we run out of offsets, extend the range
            possible_offsets = set(range(-5, 6)) - {0} - set(c - answer for c in choices)
        
        offset = random.choice(list(possible_offsets))
        new_choice = max(1, answer + offset)
        
        if new_choice not in choices:
            choices.append(new_choice)
            possible_offsets.remove(offset)
        else:
            # If the new_choice is already in choices, remove this offset
            possible_offsets.remove(offset)
    
    # Shuffle the choices
    # random.shuffle(choices)
    # correct_index = choices.index(answer)
    
    # options = ['A', 'B', 'C', 'D']
    # correct_option = options[correct_index]
    
    options, mc_answer, answer_counts = from_options_to_mc_answer(
        choices, answer, answer_counts, option_letters
    )
    
    return options, mc_answer, answer_counts


class ObjCountQAGenerator(BaseQAGenerator):
    def get_default_args(self):
        return {
            'question_template': "OBJ_COUNT_FRAME_TEMPLATE",   # changed
            'num_subsample': 6,
            'question_type': 'object_counting_frame',          # changed (distinct type)
            'output_filename_prefix': 'qa_obj_counting_frame'  # changed (distinct file)
        }

    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """
        Generate frame-wise object count QA pairs.
        For each frame, count how many instances of each category are visible in that frame
        (based on bboxes_2d -> instance_id -> category), and create a question per category
        with count > 1 (to avoid trivial singletons).
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

        # Iterate frames in order
        frames_sorted = sorted(frames, key=lambda f: f['frame_id'])

        for rank, f in enumerate(frames_sorted):
            frame_id = f.get('frame_id')
            b2d = f.get('bboxes_2d', [])
            if not b2d:
                continue

            # Count visible instances per category in THIS frame
            # Use a set per category so duplicate 2D detections for the same instance don't inflate counts
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

            # For each category with >1 visible instance in this frame, create a QA
            for category, iid_set in cat_to_iids.items():
                visible_count = len(iid_set)
                if visible_count <= 1:
                    continue

                ground_truth = visible_count

                # Generate balanced multiple-choice options and letter
                options_raw, mc_answer, self.answer_counts = generate_choices(
                    ground_truth, self.answer_counts, self.option_letters
                )

                # Keep "A. ..." formatting like your original file
                options_printable = [
                    f"A. {options_raw[0]}",
                    f"B. {options_raw[1]}",
                    f"C. {options_raw[2]}",
                    f"D. {options_raw[3]}",
                ]

                # Frame-aware wording (safe even if template ignores frame_description)
                frame_desc = f"frame {rank + 1} of {total_frames_in_meta}"
                question_text = self.question_template.format(
                    category=category,
                    frame_description=frame_desc
                )

                qa = {
                    # "id" will be assigned by the base class run method
                    'dataset': self.args.dataset,
                    "scene_name": scene_name,
                    "question_type": self.args.question_type,
                    "video_path": video_path,
                    "frame_indices": [frame_id],  # frame-wise
                    'category': category,
                    "question": question_text,
                    "options": options_printable,
                    "ground_truth": ground_truth,
                    "mc_answer": mc_answer
                }
                scene_qa_list.append(qa)

        return scene_qa_list



if __name__ == '__main__':
    # Removed argparse logic, handled by BaseQAGenerator
    generator = ObjCountQAGenerator()
    generator.run()
