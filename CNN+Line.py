import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import os

### BEV 변환 (Bird’s Eye View)
def get_perspective_matrices(frame):
    h, w = frame.shape[:2]
    
    # 도로 ROI (사다리꼴) - 영상 해상도에 맞게 수치 조정 필요
    src = np.float32([
        [w*0.45, h*0.65],   # 좌상
        [w*0.55, h*0.65],   # 우상
        [w*0.85, h*0.95],   # 우하
        [w*0.15, h*0.95]    # 좌하
    ])

    # BEV에서의 직사각형 좌표
    dst = np.float32([
        [w*0.2, 0],
        [w*0.8, 0],
        [w*0.8, h],
        [w*0.2, h]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def warp_perspective(frame, M, size):
    return cv2.warpPerspective(frame, M, size, flags=cv2.INTER_LINEAR)

### Rain Removal CNN 모델 정의
def build_derain_model(input_shape=(None, None, 3)):
    inputs = Input(shape=input_shape)

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)

    x = layers.Conv2D(128, (3,3), padding='same', dilation_rate=2, activation='relu')(x)

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    outputs = layers.Conv2D(3, (3,3), padding='same', activation='sigmoid')(x)  # [0,1] 정규화 출력

    return Model(inputs, outputs, name="DerainCNN")

### edge검출
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


if __name__ == "__main__":  
     # 1. 데이터 로드
    # trainX, trainY = load_image_pairs("data/rainy", "dataset/clean")
    # testX, testY = load_image_pairs("data/rainy", "dataset/clean")

    # print("Train:", trainX.shape, trainY.shape)

    # 2. 모델 빌드
    # model = build_derain_model()
    # model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
    #              loss="mae",   # L1 loss (픽셀 차이 최소화)
    #              metrics=["mse"])

    # 3. 학습
    # history = model.fit(trainX, trainY,
    #                    validation_data=(testX, testY),
    #                    epochs=50,
    #                    batch_size=8)

    # 4. 모델 저장
    # model.save("derain.h5")

    # CNN 모델 빌드
    #model = tf.keras.models.load_model("derain.h5")

    # CNN 모델 (학습 진행 후 156번 줄 사용)
    model = build_derain_model()
    model.summary()

    cap = cv2.VideoCapture("challenge_video.mp4")

    # 초기 프레임 → 투시 변환 행렬 계산
    ret, frame = cap.read()
    if not ret:
        print("영상 로드 실패")
        exit()
    h, w = frame.shape[:2]
    M, Minv = get_perspective_matrices(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. CNN (빗줄기 제거)
        pre_norm = frame / 255.0
        cnn_in = np.expand_dims(pre_norm, axis=0)
        cnn_out = model.predict(cnn_in)
        cnn_out = np.squeeze(cnn_out)
        cnn_out = (cnn_out * 255).astype(np.uint8)

        # 2.canny edge검출
        edge = preprocess_frame(frame)
        # 3. BEV 변환
        bev_img = warp_perspective(edge, M, (w, h))

        # 4. Sliding Window
        lane_overlay_bev = sliding_window_fit(bev_img)

        # 5. 원본 좌표계로 역투시 변환
        lane_overlay_original = cv2.warpPerspective(lane_overlay_bev, Minv, (w, h))

        # 6. 원본 영상과 합성
        result = cv2.addWeighted(frame, 1.0, lane_overlay_original, 0.6, 0)

        cv2.imshow("Lane Detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

 



