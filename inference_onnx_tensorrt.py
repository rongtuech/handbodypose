import torch
from glob import glob
from custom_augmentation import InferenceTransformation
from utils.utils_image import draw_poses_for_coco, draw_poses_for_optical_flow, draw_pose
from setting import body_edges, hand_edges
from model.mini_model import OpenPoseLightning
# from fast_inference.onnx_inference import PoseDetectionONNX
from fast_inference.onnx_with_tensorrt import PoseDetectionONNXTensorRT
import argparse
import cv2
import os
import tqdm
import numpy as np
from pose import Pose
from utils.utils_mediapipe import HandPoseDetector, FaceMeshDetector


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def preprocess_tensor(image):
    x_data = image.astype('f')
    x_data /= 255
    x_data -= 0.5
    x_data = x_data.transpose(2, 0, 1)
    return x_data

def inference_image(model, parser):
    print("image")
    # preparing
    list_image_paths = glob(os.path.join(parser.input, "*.jpg"))
    preprocess = InferenceTransformation(320, 320)
    pose_parser = Pose(image_scale=0.125)
    hand_model = HandPoseDetector(static_model=True)

    average_time = 0
    for ind, image_path in tqdm.tqdm(enumerate(list_image_paths)):
        origin_image = cv2.imread(image_path)

        height, width,_ = origin_image.shape
        ratio_scale = height / 320
        add_width = (320 - int(320 * width / height))//2
        current_time = cv2.getTickCount()
        process_image = preprocess(origin_image)
        image = preprocess_tensor(process_image.copy())

        paf, heatmap = model(image)
        paf = paf[0]
        heatmap = heatmap[0]

        pose_parser.parser_pose(paf, heatmap)
        draw_pose(process_image, pose_parser.poses_list, pose_parser.hand_window, body_edges)
        hand_images = pose_parser.get_hand_head_images(origin_image,ratio_scale, add_width)

        if hand_images is not None:
            image, _ = hand_model(hand_images)

        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        # cv2.imwrite("./_image/temp_hand_%d.jpg"%ind, image)
        if hand_images is not None:
            process_image[-200:,:100] = cv2.resize(image,(100,200))

        average_time += current_time
        cv2.putText(process_image, 'parsing time: {}'.format(current_time),
                    (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        cv2.imwrite("./_image/img_with_pose_%d.jpg" % ind, process_image)
    print("avg time: %f" % (average_time / len(list_image_paths)))

def inference_video(model, parser, is_optical_flow=False):
    print("video")
    # preparing
    list_video_paths = glob(os.path.join(parser.input, "*.mp4"))
    FIX_SIZE = 320
    preprocess = InferenceTransformation(FIX_SIZE, FIX_SIZE)
    mean_time = 0
    pose_parser = Pose(image_scale=0.125)
    hand_model = HandPoseDetector(static_model= True)
    lk_params = dict(winSize=(100, 100),
                     maxLevel=10,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    for ind, video_path in tqdm.tqdm(enumerate(list_video_paths)):
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        height, width,_ = image.shape
        ratio_scale = height / FIX_SIZE
        add_width = (FIX_SIZE - int(FIX_SIZE * width / height))//2
        count = 0
        video = cv2.VideoWriter('_image/video_with_hand_optical%d.avi'%ind, cv2.VideoWriter_fourcc(*'DIVX'), 30, (FIX_SIZE, FIX_SIZE))

        old_gray = None
        num_frame = 0
        while success:
            success, origin_image = vidcap.read()
            if success:
                current_time = cv2.getTickCount()
                # origin_image = cv2.rotate(origin_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                process_image = preprocess(origin_image)
                if is_optical_flow and 0<count <4 and len(pose_parser.poses_list) >0:
                    visible_mask = pose_parser.poses_list[0][:,2]>0
                    visible_point = pose_parser.poses_list[0][visible_mask][:,0:2] # get only coord
                    current_gray = cv2.cvtColor(process_image, cv2.COLOR_BGR2GRAY)
                    visible_point = np.expand_dims(visible_point.astype(np.float32), axis = 1)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, current_gray, visible_point,
                                                           None, **lk_params)
                    old_gray = current_gray.copy()
                    visible_point[st==1] = p1[st==1]
                    visible_point = np.squeeze(visible_point, axis=1)
                    pose_parser.poses_list[0][np.where(visible_mask),:-1] = visible_point.copy()
                    pose_parser.get_hand_head_window()
                else:
                    image = preprocess_tensor(process_image.copy())
                    # image = image.cuda()
                    paf, heatmap = model(image)
                    paf = paf.reshape((38,40,40))
                    heatmap = heatmap.reshape((19,40,40))

                    pose_parser.parser_pose(paf, heatmap)
                    old_gray = cv2.cvtColor(process_image, cv2.COLOR_BGR2GRAY)
                    count = 0

                draw_pose(process_image, pose_parser.poses_list, pose_parser.hand_window, body_edges)
                hand_images = pose_parser.get_hand_head_images(origin_image,ratio_scale, add_width)

                if hand_images is not None:
                    hand_images,_ = hand_model(hand_images)
                current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()

                if hand_images is not None:
                    process_image[-200:, :100] =cv2.resize(hand_images, (100, 200))
                if mean_time == 0:
                    mean_time = current_time
                else:
                    mean_time = mean_time * 0.95 + current_time * 0.05
                cv2.putText(process_image, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                            (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                video.write(process_image)
            count += 1
            num_frame+=1
        vidcap.release()
        video.release()
        # print(count)
        print(int(1/mean_time*10))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='inference params',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="All params for load model and inference images")
    parser.add_argument('-w', '--weight', type=str, default="./_model/bodypose_light.onnx",
                        help=r'trained model path')
    parser.add_argument('-i', '--input', type=str, default="real_data",
                        help=r'input image folder')
    parser.add_argument('-o', '--output', type=str, default="_image",
                        help=r'output image folder')
    parser.add_argument('-v', "--is_video", type=bool, default=True)
    parser.add_argument('-wh', "--with_hand", type=bool, default= True)
    parser.add_argument('-im', "--inference_model", type=str, default="tensorrt")
    parser = parser.parse_args()

    if parser.inference_model == "onnx":
        # model = PoseDetectionONNX(parser.weight)
        pass
    elif parser.inference_model == "tensorrt":
        model = PoseDetectionONNXTensorRT(parser.weight, "./openlight.trt")
    else:
        print("wrong inference name")
        exit()

    if parser.is_video:
        inference_video(model, parser, is_optical_flow=True)
    else:
        inference_image(model, parser)

    if parser.inference_model == "tensorrt":
        model.close_tensorrt()
