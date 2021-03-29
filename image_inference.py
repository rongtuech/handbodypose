from parse_poses import *
from utils.utils_image import draw_poses_for_coco
import torch
from glob import glob
from custom_augmentation import InferenceTransformation
from model.mini_model import PoseEstimationWithMobileNet
import argparse
import cv2
import os
import tqdm

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
    preprocess = InferenceTransformation(368, 368)

    average_time = 0
    for ind, image_path in tqdm.tqdm(enumerate(list_image_paths)):
        origin_image = cv2.imread(image_path)

        current_time = cv2.getTickCount()
        origin_image = preprocess(origin_image)
        image = torch.Tensor(preprocess_tensor(origin_image.copy())).unsqueeze(0)

        paf, heatmap = model(image)
        paf = paf.detach().numpy()[0]
        heatmap = heatmap.detach().numpy()[0]

        poses_2d = parse_poses((paf, heatmap), 0.125)
        draw_poses_for_coco(origin_image, poses_2d)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()

        average_time += current_time
        cv2.putText(origin_image, 'parsing time: {}'.format(current_time),
                    (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        cv2.imwrite("./_image/img_with_pose_%d.jpg" % ind, origin_image)
    print("avg time: %f" % (average_time / len(list_image_paths)))

def inference_video(model, parser):
    print("video")
    # preparing
    list_video_paths = glob(os.path.join(parser.input, "*.mp4"))
    FIX_SIZE = 320
    preprocess = InferenceTransformation(FIX_SIZE, FIX_SIZE)

    mean_time = 0
    for ind, video_path in tqdm.tqdm(enumerate(list_video_paths)):
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        height, width,_ = image.shape
        count = 0
        video = cv2.VideoWriter('_image/video_no%d.avi'%ind, cv2.VideoWriter_fourcc(*'DIVX'), 30, (FIX_SIZE, FIX_SIZE))
        while success:
            success, origin_image = vidcap.read()
            if success:
                current_time = cv2.getTickCount()
                origin_image = cv2.rotate(origin_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                origin_image = preprocess(origin_image)
                image = torch.Tensor(preprocess_tensor(origin_image.copy())).unsqueeze(0)

                paf, heatmap = model(image)
                paf = paf.detach().numpy()[0]
                heatmap = heatmap.detach().numpy()[0]

                poses_2d = parse_poses((paf, heatmap), 0.125)
                draw_poses_for_coco(origin_image, poses_2d)
                current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()

                if mean_time == 0:
                    mean_time = current_time
                else:
                    mean_time = mean_time * 0.95 + current_time * 0.05
                cv2.putText(origin_image, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                            (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                if count > 20: # wait for the process is stable after loading the images.
                    video.write(origin_image)
            count += 1

        video.release()
        print(count)
        print(mean_time)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='inference params',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="All params for load model and inference images")
    parser.add_argument('-w', '--weight', type=str, default="./_model/model_simpleepoch_2_loss_0.0029.pth",
                        help=r'trained model path')
    parser.add_argument('-i', '--input', type=str, default="real_data",
                        help=r'input image folder')
    parser.add_argument('-o', '--output', type=str, default="_image",
                        help=r'output image folder')
    parser.add_argument('-v', "--is_video", type=bool, default=True)
    parser = parser.parse_args()

    model = PoseEstimationWithMobileNet()
    model.load_state_dict(torch.load(parser.weight))

    if parser.is_video:
        inference_video(model, parser)
    else:
        inference_image(model, parser)
