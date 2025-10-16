import torch
import torch.nn as nn
import cv2
import numpy as np

# --- RRBC 모델 정의 ---

# --- 1. Squeeze-and-Excitation (SE) 블럭 정의 ---
# 채널별로 중요한 특징에 가중치 부여
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze: 특징맵을 전역 평균 풀링을 통해 채널별 대표값으로 압축
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation: 압축된 정보를 바탕으로 채널별 중요도(가중치)를 학습
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() # 가중치를 0~1 사이 값으로 만듦
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze & Excitation
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 원본 특징맵에 학습된 중요도를 곱하여 중요한 특징을 강조
        return x * y.expand_as(x)


# --- 2. RRBC의 핵심 CNN 블록 정의 ---
# 논문에서 반복적으로 사용되는 CNN 파트
# Dilated Convolution을 포함하여 이미지의 넓은 영역에서 특징 추출
class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        # 3x3 크기의 일반 합성곱 레이어
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # 3x3 크기의 팽창된(Dilated) 합성곱 레이어. rate=2
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.relu = nn.ReLU()

        # SE 블럭 인스턴스화
        self.se = SEBlock(channels)

    def forward(self, x):
        # 잔차 연결(Residual Connection)을 위해 원본 입력을 저장
        residual = x
        
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        # 잔차와 합치기 전에 SE 블럭 통과
        out = self.se(out)

        # 모델의 출력에 원본 입력을 더함 (핵심!!!)
        out += residual
        return self.relu(out)

# --- 3. ConvLSTM 셀 정의 ---
# 이미지 데이터(2D)를 처리할 수 있는 LSTM 셀
# 이전 단계(Stage)의 '기억(hidden_state)'을 다음 단계로 전달하는 핵심 역할
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # 입력과 이전 기억을 받아 4개의 게이트 값을 한 번에 계산
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, prev_hidden_state):
        # 이전 단계의 기억이 없다면(첫 단계) 0으로 초기화
        if prev_hidden_state is None:
            h_prev = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device)
            c_prev = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device)
        else:
            h_prev, c_prev = prev_hidden_state

        # 입력과 이전 기억(h_prev)을 채널 방향으로 합함
        combined = torch.cat([x, h_prev], dim=1)
        
        # 4개의 게이트(input, forget, output, gate) 값을 계산
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        # LSTM의 핵심 연산 수행
        c_cur = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        h_cur = torch.sigmoid(o) * torch.tanh(c_cur)

        return h_cur, c_cur

# --- 4. 최종 RRBC 전체 모델 조립 ---
# 위에서 만든 부품들(RRB, ConvLSTM)을 조립하여 완전한 RRBC 네트워크 형성
# 훈련시키게 될 최종 모델
class RRBC_Net(nn.Module):
    def __init__(self, in_channels=3, feature_channels=64, num_stages=3):
        super(RRBC_Net, self).__init__()
        self.num_stages = num_stages

        # 입력 이미지를 딥러닝이 처리할 특징(feature)으로 변환하는 첫 레이어
        self.conv_in = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)

        # 핵심 부품들 인스턴스화
        self.rrb = RecurrentResidualBlock(channels=feature_channels)
        self.lstm = ConvLSTMCell(input_dim=feature_channels, hidden_dim=feature_channels, kernel_size=3)

        # 처리된 특징을 다시 이미지(빗줄기)로 변환하는 마지막 레이어
        self.conv_out = nn.Conv2d(feature_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 0. 원본 비 오는 이미지 저장 (마지막에 더하기 위함)
        original_image = x

        # 1. 입력 이미지를 특징으로 변환
        features = self.relu(self.conv_in(x))

        # 2. LSTM의 '기억'을 초기화
        hidden_state = None

        # 3. RNN 구조: 정해진 단계(Stage)만큼 반복적으로 비를 제거
        for _ in range(self.num_stages):
            # CNN 블록을 통과하여 이미지 특징 추출
            features = self.rrb(features)
            # LSTM 셀을 통과하며 이전 단계의 기억을 활용
            h, c = self.lstm(features, hidden_state)
            hidden_state = (h, c)
            # LSTM의 출력을 다음 단계의 CNN 입력으로 사용
            features = h
        
        # 4. 최종적으로 예측된 '빗줄기' 레이어
        rain_layer = self.conv_out(features)
        
        # 5. 원본 이미지에서 예측된 빗줄기를 빼서 깨끗한 이미지 획득
        derained_image = original_image - rain_layer

        return derained_image

# --- 비디오 처리를 위한 실행 부분 ---
if __name__ == '__main__':
    # 0. GPU 사용 가능 여부 확인 및 모델 이동 (GPU 불가 시 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. 모델 생성 및 추론 모드 설정
    model = RRBC_Net(num_stages=3)
    model.load_state_dict(torch.load('rrbc_model_trained.pth', map_location=device))
    model.to(device)
    model.eval()

    print(f"Using device: {device}")

    # 2. 비디오 파일 열기
    video_path = 'rainy_video.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        exit()

    # 3. 비디오의 각 프레임을 순회하며 처리
    while cap.isOpened():
        ret, frame = cap.read() # 프레임 하나 읽기 BGR 순서
        if not ret:
            print("비디오의 끝에 도달했거나 오류가 발생했습니다.")
            break

        # --- 프레임 전처리 ---
        # 원본 프레임의 크기를 모델 입력에 맞게 조정 (예: 480x640)
        # (성능을 위해 원본 크기보다 작게 조절하는 것이 일반적)
        h, w, _ = frame.shape
        input_size = (480, 320) # (가로, 세로)
        frame_resized = cv2.resize(frame, input_size)
        
        # OpenCV(BGR) 이미지를 PyTorch가 인식할 RGB 순서로 변환
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # 이미지를 PyTorch 텐서로 변환 (NumPy -> Tensor)
        image_tensor = torch.from_numpy(frame_rgb.transpose((2, 0, 1))).float() / 255.0
        input_tensor = image_tensor.unsqueeze(0).to(device) # GPU로 데이터 이동

        # --- 모델 추론 ---
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # --- 결과 후처리 ---
        # 출력 텐서를 다시 화면에 표시할 이미지로 변환 (Tensor -> NumPy)
        output_np = output_tensor.squeeze(0).cpu().detach().numpy()
        output_np = (output_np.transpose((1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)

        # --- 화면에 결과 표시 ---
        # 원본(리사이즈된) 영상과 결과 영상을 나란히 붙여서 보여주기
        combined_output = np.hstack((frame_resized, output_np))
        cv2.imshow('Rain Removal (Original vs. RRBC Output)', combined_output)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. 자원 해제
    cap.release()
    cv2.destroyAllWindows()
