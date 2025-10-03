import torch
import torch.nn as nn
import cv2  # OpenCV 라이브러리
import numpy as np

# --- 1. 이전에 작성했던 RRBC 모델 코드 (수정 없음) ---
class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, prev_hidden_state):
        if prev_hidden_state is None:
            # .to(x.device)를 추가하여 입력텐서와 동일한 장치(CPU 또는 GPU)를 사용
            h_prev = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device)
            c_prev = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device)
        else:
            h_prev, c_prev = prev_hidden_state

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        c_cur = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        h_cur = torch.sigmoid(o) * torch.tanh(c_cur)
        return h_cur, c_cur

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

# --- 2. 실제 이미지를 불러와 모델을 실행하는 부분 ---
if __name__ == '__main__':
    # 1. 모델 생성
    model = RRBC_Net(num_stages=3)
    # 모델을 추론 모드로 설정 (훈련할 것이 아니므로)
    model.eval()

    # 2. 이미지 불러오기 및 전처리
    # OpenCV로 이미지 읽기 (결과는 NumPy 배열)
    image_path = 'rainy_image.jpg'
    try:
        image_np = cv2.imread(image_path)  
        # 이미지를 PyTorch가 다루는 형식으로 변환
        # (H, W, C) -> (C, H, W), BGR -> RGB, 0-255 -> 0.0-1.0
        image_tensor = torch.from_numpy(image_np.transpose((2, 0, 1))).float() / 255.0
        # 모델은 (Batch, C, H, W) 형태의 4차원 텐서를 입력으로 받으므로 차원 추가
        input_tensor = image_tensor.unsqueeze(0)
    except Exception as e:
        print(f"이미지를 불러오는 데 실패했습니다: {e}")
        print("코드와 같은 폴더에 'rainy_image.jpg' 파일이 있는지 확인하세요.")
        exit()


    # 3. 모델에 이미지 입력하여 추론 실행
    print("훈련되지 않은 모델에 이미지를 입력합니다...")
    with torch.no_grad(): # 기울기 계산을 하지 않아 메모리 사용량과 계산 속도를 개선
        output_tensor = model(input_tensor)
    print("추론 완료.")

    # 4. 출력 결과 후처리 및 시각화
    # PyTorch 텐서를 다시 시각화를 위한 NumPy 배열로 변환
    # (Batch, C, H, W) -> (C, H, W) -> (H, W, C), 0.0-1.0 -> 0-255
    output_np = output_tensor.squeeze(0).cpu().detach().numpy()
    output_np = (output_np.transpose((1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)

    # 원본 이미지와 결과 이미지를 화면에 표시
    cv2.imshow('Original Rainy Image', image_np)
    cv2.imshow('Untrained Model Output', output_np)

    print("아무 키나 누르면 창이 닫힙니다.")
    cv2.waitKey(0) # 사용자가 키를 누를 때까지 대기
    cv2.destroyAllWindows()