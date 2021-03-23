from utils.utils_common import FileTool
import json
import tqdm
init_file_path = "../../data/coco_bodypose/person_keypoints_train2017.json"
# init_file_path = "../../data/coco_bodypose/person_keypoints_val2017.json"
output_file_path = "../../data/coco_bodypose"
ids = []
file_infos = {}
annotations =  {}

with open(init_file_path) as json_file:
    data = json.load(json_file)
    images = data["images"]
    raw_annotaion = data['annotations']

    print("write image")
    for ele in tqdm.tqdm(images):
        if ele["id"] not in ids:
            ids.append(ele["id"])
        else:
            print("shit already")
        # file_infos[ele["id"]] = {
        #     "id": ele["id"],
        #     "w": ele["width"],
        #     "h": ele["height"],
        #     'name': ele["file_name"]
        # }

    print("write annotations")
    for ele in tqdm.tqdm(raw_annotaion):
        if ele["image_id"] not in ids:
            print("dmdmdmmd")
        if ele["image_id"] in annotations:
            annotations[ele["image_id"]].append({
                "image_id": ele["image_id"],
                "is_crowd": ele["iscrowd"],
                "area": ele["area"],
                "keypoints": ele["keypoints"],
                "num_keypoints": ele["num_keypoints"]
            })
        else:
            annotations[ele["image_id"]] = [{
                "image_id": ele["image_id"],
                "is_crowd": ele["iscrowd"],
                "area": ele["area"],
                "keypoints": ele["keypoints"],
                "num_keypoints": ele["num_keypoints"]
            }]

import os

print("write output")
# FileTool.writePickle(os.path.join(output_file_path, "val_ids.pkl"), ids)
# FileTool.writePickle(os.path.join(output_file_path, "val_file_infos.pkl"), file_infos)
FileTool.writePickle(os.path.join(output_file_path, "annotation_ids.pkl"), annotations)

