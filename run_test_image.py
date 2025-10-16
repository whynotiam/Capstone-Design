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

# --- 실제 이미지를 불러와 모델을 실행하는 부분 ---
if __name__ == '__main__':
    # 0. GPU 사용 가능 여부 확인 및 모델 이동 (GPU 불가 시 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. 모델 생성 및 추론 모드 설정
    model = RRBC_Net(num_stages=3)
    # 훈련된 모델 가중치 엔진을 불러와 모델에 장착
    model.load_state_dict(torch.load('rrbc_model_trained.pth', map_location=device))
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    # 2. 이미지 불러오기 및 전처리
    # OpenCV로 이미지 읽기 (결과는 NumPy 배열)
    image_path = 'rainy_image.jpg'
    try:
        image_np_bgr = cv2.imread(image_path)

        image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)

        # 이미지를 PyTorch가 다루는 형식으로 변환
        # (H, W, C) -> (C, H, W), BGR -> RGB, 0-255 -> 0.0-1.0
        image_tensor = torch.from_numpy(image_np_rgb.transpose((2, 0, 1))).float() / 255.0
        # 모델은 (Batch, C, H, W) 형태의 4차원 텐서를 입력으로 받으므로 차원 추가
        input_tensor = image_tensor.unsqueeze(0).to(device)
    except Exception as e:
        print(f"이미지를 불러오는 데 실패했습니다: {e}")
        print("코드와 같은 폴더에 'rainy_image.jpg' 파일이 있는지 확인하세요.")
        exit()


    # 3. 모델에 이미지 입력하여 추론 실행
    with torch.no_grad(): # 기울기 계산을 하지 않아 메모리 사용량과 계산 속도를 개선
        output_tensor = model(input_tensor)
    print("추론 완료.")

    # 4. 출력 결과 후처리 및 시각화
    # PyTorch 텐서를 다시 시각화를 위한 NumPy 배열로 변환
    # (Batch, C, H, W) -> (C, H, W) -> (H, W, C), 0.0-1.0 -> 0-255
    output_np = output_tensor.squeeze(0).cpu().detach().numpy()
    output_np_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    output_np_final = (output_np_bgr * 255.0).clip(0, 255).astype(np.uint8)

    # 원본 이미지와 결과 이미지를 화면에 표시
    cv2.imshow('Original Rainy Image', image_np_bgr)
    cv2.imshow('Untrained Model Output', output_np_final)

    print("아무 키나 누르면 창이 닫힙니다.")
    cv2.waitKey(0) # 사용자가 키를 누를 때까지 대기
    cv2.destroyAllWindows()
