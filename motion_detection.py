#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
使用OpenCV进行视频流运动检测
'''


import cv2
from datetime import datetime


BLUR_KERNEL_P0 = (21, 21)
BLUR_KERNEL_P1 = (11, 11)
LEARNING_RATE = 0.001
MIN_CONTOUR_AREA = 625

# 使用这台电脑的第一个摄像头
feed = cv2.VideoCapture(0)
# 关闭摄像头自动曝光
feed.set(cv2.CAP_PROP_AUTO_EXPOSURE, False)
# 使用KNN背景检测
fgbg = cv2.createBackgroundSubtractorKNN()

(_, frame) = feed.read()
(FRAME_HEIGHT, FRAME_WIDTH, _) = frame.shape
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT

while True:
    (grabbed, frame) = feed.read()

    if not grabbed:
        break

    # Using YCbCr Color model to get rid of lighting condition changes,
    # got removed because of the bad result it produce.
    # frame_YCbCr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCbCr)
    # _, Cb, Cr = cv2.split(frame_hsv)
    # frame_CbCr = cv2.merge((Cb, Cr))

    # 高斯模糊，除去部分噪声
    frame_blured = cv2.GaussianBlur(frame, BLUR_KERNEL_P0, 0)
    fgmask = fgbg.apply(frame_blured, learningRate=LEARNING_RATE)

    # 同上，噪声过滤
    thresh = cv2.GaussianBlur(fgmask, BLUR_KERNEL_P1, 0)
    thresh = cv2.dilate(thresh, None, iterations=2)
    # thresh = cv2.dilate(fgmask, None, iterations=2)
    # thresh = cv2.GaussianBlur(thresh, (11, 11), 0)

    # 寻找轮廓并丢弃轮廓里面的轮廓
    (_, cnts, _) = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    area_in_motion = 0
    for cnt in cnts:
        # 去掉较小的轮廓（可能为噪声）
        area = cv2.contourArea(cnt)
        if  area < MIN_CONTOUR_AREA:
            continue

        area_in_motion += area
        # 根据轮廓绘制矩形
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 255, 255), 2)

    cv2.putText(
        frame, "Faction of frame in motion: {}".format(area_in_motion/FRAME_SIZE),
        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
    )
    cv2.putText(
        frame, datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1
    )
    # Show results
    cv2.imshow("Feed", frame)
    cv2.imshow("Thresh", thresh)

    # Wait for key 'ESC' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# That's how you exit
feed.release()
cv2.destroyAllWindows()
