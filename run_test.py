import torch
import torch.nn as nn
import cv2
import numpy as np

# --- 1. RRBC의 핵심 CNN 블록 정의 ---
# Dilated Convolution을 포함하여 이미지의 넓은 영역에서 특징 추출
class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        # 3x3 크기의 일반 합성곱 레이어
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # 3x3 크기의 팽창된(Dilated) 합성곱 레이어. rate=2
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        out += residual
        return out

# --- 2. ConvLSTM 셀 정의 ---
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, prev_hidden_state):
        if prev_hidden_state is None:
            h_prev, c_prev = torch.zeros_like(x), torch.zeros_like(x)
        else:
            h_prev, c_prev = prev_hidden_state

        combined = torch.cat([x, h_prev], dim=1)
        
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        c_cur = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        h_cur = torch.sigmoid(o) * torch.tanh(c_cur)

        return h_cur, c_cur

# --- 3. 최종 RRBC 전체 모델 조립 ---
# 위에서 만든 부품들(RRB, ConvLSTM)을 조립하여 완전한 RRBC 네트워크 형성
# 훈련시키게 될 최종 모델
class RRBC_Net(nn.Module):
    def __init__(self, in_channels=3, feature_channels=64, num_stages=3):
        super(RRBC_Net, self).__init__()
        self.num_stages = num_stages

        self.conv_in = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)

        self.rrb = RecurrentResidualBlock(channels=feature_channels)
        self.lstm = ConvLSTMCell(input_dim=feature_channels, hidden_dim=feature_channels, kernel_size=3)

        self.conv_out = nn.Conv2d(feature_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        original_image = x

        x = self.conv_in(x)

        hidden_state = None

        for _ in range(self.num_stages):
            x = self.rrb(x)
            h, c = self.lstm(x, hidden_state)
            hidden_state = (h, c)
            x = h
        
        rain_layer = self.conv_out(x)
        
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
        ret, frame = cap.read() # 프레임 하나 읽기
        if not ret:
            print("비디오의 끝에 도달했거나 오류가 발생했습니다.")
            break

        # --- 프레임 전처리 ---
        # 원본 프레임의 크기를 모델 입력에 맞게 조정 (예: 480x640)
        # (성능을 위해 원본 크기보다 작게 조절하는 것이 일반적)
        h, w, _ = frame.shape
        input_size = (480, 320) # (가로, 세로)
        frame_resized = cv2.resize(frame, input_size)
        
        # 이미지를 PyTorch 텐서로 변환 (NumPy -> Tensor)
        image_tensor = torch.from_numpy(frame_resized.transpose((2, 0, 1))).float() / 255.0
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



