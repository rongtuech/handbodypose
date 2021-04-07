import numpy as np
try:
    from pose_extractor import extract_poses
    print("using c++")
except:
    from legacy_pose_extractor import extract_poses

AVG_PERSON_HEIGHT = 180

# pelvis (body center) is missing, id == 2
map_id_to_panoptic = [1, 0, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17, 18]

limbs = [[18, 17, 1],
         [16, 15, 1],
         [5, 4, 3],
         [8, 7, 6],
         [11, 10, 9],
         [14, 13, 12]]


def get_root_relative_poses(inference_results):
    paf_map, heatmap = inference_results

    upsample_ratio = 4
    found_poses = extract_poses(heatmap[0:-1], paf_map, upsample_ratio)[0]
    # scale coordinates to features space
    found_poses[:, 0:-1:3] /= upsample_ratio
    found_poses[:, 1:-1:3] /= upsample_ratio

    poses_2d = []
    num_kpt_panoptic = 19
    num_kpt = 18
    for pose_id in range(found_poses.shape[0]):
        if found_poses[pose_id, 3] == -1:  # skip pose if does not found neck
            continue
        pose_2d = np.ones(num_kpt_panoptic * 3 + 1, dtype=np.float32) * -1  # +1 for pose confidence
        for kpt_id in range(num_kpt):
            if found_poses[pose_id, kpt_id * 3] != -1:
                x_2d, y_2d = found_poses[pose_id, kpt_id * 3:kpt_id * 3 + 2]
                conf = found_poses[pose_id, kpt_id * 3 + 2]
                pose_2d[map_id_to_panoptic[kpt_id] * 3] = x_2d  # just repacking
                pose_2d[map_id_to_panoptic[kpt_id] * 3 + 1] = y_2d
                pose_2d[map_id_to_panoptic[kpt_id] * 3 + 2] = conf
        pose_2d[-1] = found_poses[pose_id, -1]
        poses_2d.append(pose_2d)

    return np.array(poses_2d)


previous_poses_2d = []


def parse_poses(inference_results, input_scale):
    """
    parse inferece (paf, heatmap)
    :param inference_results:
    :param input_scale: scale of image and heatmap
    :return:list of pose = tuple of  (keypointsx3) + pose confidence)
    """
    global previous_poses_2d
    poses_2d = get_root_relative_poses(inference_results)
    # return poses_2d
    poses_2d_scaled = []
    poses_prop = []
    for pose_2d in poses_2d:
        num_kpt = (pose_2d.shape[0] - 1) // 3
        pose_2d_scaled = np.ones((num_kpt, 3), dtype=np.float32) * -1  # +1 for pose confidence
        for kpt_id in range(num_kpt):
            if pose_2d[kpt_id * 3] != -1:
                pose_2d_scaled[kpt_id * 3] = int(pose_2d[kpt_id * 3] / input_scale)
                pose_2d_scaled[kpt_id * 3 + 1] = int(pose_2d[kpt_id * 3 + 1] / input_scale)
                pose_2d_scaled[kpt_id * 3 + 2] = pose_2d[kpt_id * 3 + 2]
        poses_2d_scaled.append(pose_2d_scaled)
        poses_prop.append(pose_2d[-1])

    return poses_2d_scaled, poses_prop
