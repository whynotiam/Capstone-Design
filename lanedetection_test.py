#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

### BEV 변환 (Bird’s Eye View)
def get_perspective_matrices(frame):
    h, w = frame.shape[:2]
    
    # 도로 ROI (사다리꼴) - challenge_video.mp4에 맞게 튜닝
    src = np.float32([
        [w*0.43, h*0.62],   # 좌상
        [w*0.57, h*0.62],   # 우상
        [w*0.95, h*0.95],   # 우하
        [w*0.05, h*0.95]    # 좌하
    ])

    # BEV에서의 직사각형 좌표
    dst = np.float32([
        [w*0.2, 0],
        [w*0.8, 0],
        [w*0.8, h],
        [w*0.2, h]
    ])

    # 변환 행렬
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def warp_perspective(frame, M, size):
    return cv2.warpPerspective(frame, M, size, flags=cv2.INTER_LINEAR)

### 전처리:  엣지 검출
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

### 다중 슬라이딩 윈도우 차선 검출
def sliding_window_fit(binary_img, nwindows=9, margin=100, minpix=50):
    histogram = np.sum(binary_img[binary_img.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = int(binary_img.shape[0] // nwindows)
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    out_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

    for window in range(nwindows):
        win_y_low = binary_img.shape[0] - (window + 1) * window_height
        win_y_high = binary_img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) > 0 and len(lefty) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = None
    if len(rightx) > 0 and len(righty) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = None

    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])

    if left_fit is not None:
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        cv2.polylines(out_img, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=5)

    if right_fit is not None:
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
        cv2.polylines(out_img, np.int32([pts_right]), isClosed=False, color=(255, 0, 0), thickness=5)

    return out_img

### 메인 실행
if __name__ == "__main__":
    cap = cv2.VideoCapture("challenge_video.mp4")

    # 초기 프레임으로 투시 변환 행렬 계산
    ret, frame = cap.read()
    if not ret:
        print("영상 로드 실패")
        exit()
    h, w = frame.shape[:2]
    M, Minv = get_perspective_matrices(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 다시 처음으로

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 전처리 (ROI + Canny)
        edges = preprocess_frame(frame)

        # 2. BEV 변환
        bev_binary = warp_perspective(edges, M, (w,h))

        # 3. BEV 공간에서 슬라이딩 윈도우
        lane_overlay_bev = sliding_window_fit(bev_binary)

        # 4.원본 좌표계로 역투시 변환
        lane_overlay_original = cv2.warpPerspective(lane_overlay_bev, Minv, (w,h))

        # 5. 원본 영상에 합성
        result = cv2.addWeighted(frame, 1.0, lane_overlay_original, 0.6, 0)

        cv2.imshow("Lane Detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
