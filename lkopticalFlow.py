import cv2 as cv
import numpy as np
import cv2
import os
from glob import glob

list_video_paths = glob(os.path.join("real_data", "*.mp4"))
cap = cv.VideoCapture(list_video_paths[0])

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (50,50),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

height, width, _ = old_frame.shape
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
video = cv2.VideoWriter('_image/video_lk.avi',
                        cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))
count = 0
while(1):
    ret,frame = cap.read()
    # count +=1
    # if not ret:
    #     break
    # if count == 100:
    #     p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    #     old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # elif count < 100:
    #     old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     continue
    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # if count %150==1:
    #     p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    #     mask = np.zeros_like(old_frame)
    # # calculate optical flow
    # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #
    # # Select good points
    # good_new = p1[st==1]
    # good_old = p0[st==1]
    #
    # # draw the tracks
    # for i,(new,old) in enumerate(zip(good_new,good_old)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    # img = cv2.add(frame,mask)
    #
    # video.write(img)
    #
    # # Now update the previous frame and previous points
    # old_gray = frame_gray.copy()
    # p0 = good_new.reshape(-1,1,2)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2', rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)
    prvs = next

video.release()
cap.release()