"""
The format of the qa file is as follows:
{
    "id": id,
    "scene_name": "xxxx",
    "question_type": "the question type a.k.a. the task at hand",
    "video_path": "path to the video",
    "category, positioning object": "category of positioning object",
    "category, orienting object": "category of orienting object",
    "category, querying object": "category of querying object",
    "question": "question",
    "options": "answer options",
    "ground truth": "either the float or string answer",
    "mc_answer": "letter answer",
}
"""
import math
import random
import itertools
import numpy as np
import open3d as o3d

from preprocessing.preprocess_scannet.common_utils import cal_3d_bbox_distance_between_categories, from_options_to_mc_answer

from ..base_qa_generator import BaseQAGenerator


def ambiguous(object_triple, dist_lookup, obj_bbox_info, min_dist=0.30, max_dist=5.0):
    """
    Check if the distance between the two objects makes the direction question ambiguous.
    """
    pairs = [[object_triple[0], object_triple[1]], 
             [object_triple[0], object_triple[2]], 
             [object_triple[1], object_triple[2]]]

    for category1, category2 in pairs:
        if (category1, category2) in dist_lookup:
            dist = dist_lookup[(category1, category2)]
        elif (category2, category1) in dist_lookup:
            dist = dist_lookup[(category2, category1)]
        else:
            dist = cal_3d_bbox_distance_between_categories(
                obj_bbox_info[category1], obj_bbox_info[category2]
            )
            dist_lookup[(category1, category2)] = dist
            dist_lookup[(category2, category1)] = dist

        if dist < min_dist or dist > max_dist:
            return True

    return False


def get_gt(positioning_obj, orienting_obj, querying_obj, answer_counts, option_letters):
    positioning_obj_pos = np.array([positioning_obj['centroid']])
    orienting_obj_pos = np.array([orienting_obj['centroid']])
    querying_obj_pos = np.array([querying_obj['centroid']])
    
    querying_bbox = o3d.geometry.OrientedBoundingBox(
            center=np.array(querying_obj["centroid"]).astype(np.float64).reshape(3, 1),
            R=np.array(querying_obj["normalizedAxes"]).astype(np.float64).reshape(3, 3).T,
            extent=np.array(querying_obj["axesLengths"]).astype(np.float64).reshape(3, 1)
        )
    
    vertices = np.asarray(querying_bbox.get_box_points())
    querying_points = np.concatenate([querying_obj_pos, vertices], axis=0)
    
    orienting_vec = orienting_obj_pos - positioning_obj_pos
    querying_vecs = querying_points - positioning_obj_pos
    
    def calculate_angle(v1, v2s):
        dot_products = (v1 * v2s).sum(axis=1)
        mag_v1 = np.linalg.norm(v1, axis=1)
        mag_v2s = np.linalg.norm(v2s, axis=1)
        
        angles = np.arccos(dot_products / (mag_v1 * mag_v2s))
        
        crs_products = np.cross(v1, v2s)
        
        angles = np.where(crs_products >= 0., angles, 2 * math.pi - angles)
        return np.degrees(angles)

    angles = calculate_angle(orienting_vec[:, :2], querying_vecs[:, :2])
    
    quadrant_of_centroid = (angles // 90)[0]
    quadrant_of_vertices = (angles // 90)[1:]

    if (quadrant_of_centroid != quadrant_of_vertices).sum() > 2: # if more than two vertices falls into a different quadrant compared to the centroid, skip this sample
        return "ambiguous", None, None, answer_counts
    
    ambiguity_threshold = 10
    if (90 - (angles % 90)[0]) < ambiguity_threshold:
        return "ambiguous", None, None, answer_counts

    # randomly shuffle direction options
    directions = ['front-left', 'front-right', 'back-left', 'back-right']
    
    if angles[0] >= 270:
        gt = "front-right"
    elif angles[0] >= 180:
        gt = "back-right"
    elif angles[0] >= 90:
        gt = "back-left"
    else:
        gt = "front-left"
    
    options, mc_answer, answer_counts = from_options_to_mc_answer(directions, gt, answer_counts, option_letters)

    return gt, mc_answer, options, answer_counts


class ObjRelDirV1QAGenerator(BaseQAGenerator):
    def get_default_args(self):
        return {
            'question_template': "OBJ_REL_DIRECTION_V1_FRAME_TEMPLATE",  # changed
            'num_subsample': 6,
            'question_type': 'object_rel_direction_v1_frame',             # changed (new type)
            'output_filename_prefix': 'qa_obj_rel_direction_v1_frame'     # changed (new output filename)
        }

    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """
        Generate frame-wise relative direction QA pairs.

        For each frame, consider objects that are visible exactly once (based on bboxes_2d).
        For each valid triple (positioning, orienting, querying), compute their relative
        direction relationships and generate multiple-choice questions.
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

        # Also build category -> {iid -> obj_dict} for direct metadata lookup
        cat_iid_to_obj = {}
        for category, instances in obj_bboxes.items():
            if not isinstance(instances, list):
                continue
            cat_iid_to_obj[category] = {}
            for obj in instances:
                iid = obj.get('instance_id')
                if iid is None:
                    continue
                cat_iid_to_obj[category][int(iid)] = obj

        # Iterate frames
        frames_sorted = sorted(frames, key=lambda f: f['frame_id'])
        dist_lookup = {}

        for rank, f in enumerate(frames_sorted):
            frame_id = f.get('frame_id')
            b2d = f.get('bboxes_2d', [])
            if not b2d:
                continue

            # Count categories with exactly one visible instance
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

            unique_cats = [c for c, s in cat_to_iids.items() if len(s) == 1]
            if len(unique_cats) < 3:
                continue

            # Subsample object triples if too many
            object_triples = list(itertools.combinations(unique_cats, 3))
            if len(object_triples) > self.args.num_subsample:
                object_triples = random.sample(object_triples, self.args.num_subsample)

            for object_triple in object_triples:
                # Skip ambiguous triples based on distance thresholds
                if ambiguous(object_triple, dist_lookup, obj_bboxes):
                    continue

                # Generate 3 permutations for each triple
                object_triple_permutations = [
                    (object_triple[0], object_triple[1], object_triple[2]),
                    (object_triple[0], object_triple[2], object_triple[1]),
                    (object_triple[1], object_triple[0], object_triple[2])
                ]

                for positioning_obj, orienting_obj, querying_obj in object_triple_permutations:
                    pos_iid = next(iter(cat_to_iids[positioning_obj]))
                    ori_iid = next(iter(cat_to_iids[orienting_obj]))
                    qry_iid = next(iter(cat_to_iids[querying_obj]))

                    pos_meta = cat_iid_to_obj[positioning_obj].get(pos_iid)
                    ori_meta = cat_iid_to_obj[orienting_obj].get(ori_iid)
                    qry_meta = cat_iid_to_obj[querying_obj].get(qry_iid)
                    if not all([pos_meta, ori_meta, qry_meta]):
                        continue

                    gt, mc_answer, options, self.answer_counts = get_gt(
                        positioning_obj=pos_meta,
                        orienting_obj=ori_meta,
                        querying_obj=qry_meta,
                        answer_counts=self.answer_counts,
                        option_letters=self.option_letters
                    )
                    if gt == "ambiguous":
                        continue

                    frame_desc = f"frame {rank + 1} of {total_frames_in_meta}"
                    question_text = self.question_template.format(
                        positioning_object=positioning_obj,
                        orienting_object=orienting_obj,
                        querying_object=querying_obj,
                        frame_description=frame_desc
                    )

                    qa = {
                        'dataset': self.args.dataset,
                        "scene_name": scene_name,
                        "video_path": video_path,
                        "frame_indices": [frame_id],  # NEW: frame-wise link
                        "question_type": self.args.question_type,
                        "category, positioning object": positioning_obj,
                        "category, orienting object": orienting_obj,
                        "category, querying object": querying_obj,
                        "question": question_text,
                        "options": [
                            f"A. {options[0]}",
                            f"B. {options[1]}",
                            f"C. {options[2]}",
                            f"D. {options[3]}",
                        ],
                        "ground_truth": gt,
                        "mc_answer": mc_answer,
                    }
                    scene_qa_list.append(qa)

        return scene_qa_list

if __name__ == '__main__':
    generator = ObjRelDirV1QAGenerator()
    generator.run()