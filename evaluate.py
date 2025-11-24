import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np

# --- 1. PSNR/SSIM 계산을 위한 라이브러리 ---
# pip install scikit-image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- 2. RRBC 모델 정의 (train.py에서 복사) ---
# --- 2-1. Squeeze-and-Excitation (SE) 블록 정의 ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() 
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- 2-2. SE 블록이 포함된 Recurrent Residual Block ---
class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        out = self.se(out)
        
        out += residual
        return self.relu(out) 

# --- 2-3. ConvLSTM 셀 정의 ---
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, prev_hidden_state):
        if prev_hidden_state is None:
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

# --- 2-4. 최종 RRBC 전체 모델 조립 ---
class RRBC_Net(nn.Module):
    def __init__(self, in_channels=3, feature_channels=64, num_stages=3):
        super(RRBC_Net, self).__init__()
        self.num_stages = num_stages
        self.conv_in = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)
        self.rrb = RecurrentResidualBlock(channels=feature_channels)
        self.lstm = ConvLSTMCell(input_dim=feature_channels, hidden_dim=feature_channels, kernel_size=3)
        self.conv_out = nn.Conv2d(feature_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        original_image = x
        
        features = self.relu(self.conv_in(x))
        hidden_state = None

        for _ in range(self.num_stages):
            features = self.rrb(features)
            h, c = self.lstm(features, hidden_state)
            hidden_state = (h, c)
            features = h
        
        rain_layer = self.conv_out(features)
        derained_image = original_image - rain_layer
        return derained_image

# --- 3. 테스트 데이터셋 로더 ---
class RainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.rainy_dir = os.path.join(self.root_dir, 'rainy')
        self.clean_dir = os.path.join(self.root_dir, 'clean')
        self.image_files = os.listdir(self.rainy_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. rainy 이미지 이름과 경로 설정
        rainy_img_name = self.image_files[idx] # 예: 'rain-001.jpg'
        rainy_path = os.path.join(self.rainy_dir, rainy_img_name)
        
        # --- ✨ 'rain-' -> 'norain-' 이름 규칙 적용 ✨ ---
        clean_img_name = rainy_img_name.replace('rain-', 'norain-', 1)
        clean_path = os.path.join(self.clean_dir, clean_img_name)
        
        # 3. 이미지 로딩
        rainy_image = Image.open(rainy_path).convert("RGB")
        clean_image = Image.open(clean_path).convert("RGB")
        
        # 4. 전처리(transform) 적용
        if self.transform:
            rainy_image = self.transform(rainy_image)
            clean_image = self.transform(clean_image)
            
        return rainy_image, clean_image

# --- 4. 평가(Evaluation) 실행 부분 ---
if __name__ == '__main__':
    # 1. 모델 및 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RRBC_Net(num_stages=3).to(device)
    
    # 훈련된 모델 가중치('엔진') 불러오기
    model.load_state_dict(torch.load('rrbc_model_trained.pth', map_location=device))
    
    # 모델을 평가 모드로 설정! (필수)
    model.eval()

    print(f"Using device: {device}")
    print("Starting evaluation...")

    # 2. 테스트 데이터 로더 설정
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor() 
    ])
    
    test_dataset = RainDataset(root_dir='test image folder', transform=test_transform)
    # (주의: 테스트 시에는 shuffle=False가 정석)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 3. PSNR 및 SSIM 점수를 저장할 리스트 초기화
    total_psnr = 0
    total_ssim = 0
    image_count = 0

    # 4. 평가 루프
    with torch.no_grad(): # 기울기 계산 비활성화 (필수)
        for rainy_image, clean_image in test_loader:
            rainy_image = rainy_image.to(device)
            clean_image = clean_image.to(device)

            # 모델 추론
            predicted_image = model(rainy_image)

            # --- PSNR / SSIM 계산을 위한 텐서 변환 ---
            
            # (1, C, H, W) -> (C, H, W)
            clean_image_np = clean_image.squeeze(0).cpu().numpy()
            predicted_image_np = predicted_image.squeeze(0).cpu().numpy()
            
            # (C, H, W) -> (H, W, C)
            clean_image_np = np.transpose(clean_image_np, (1, 2, 0))
            predicted_image_np = np.transpose(predicted_image_np, (1, 2, 0))
            
            # 픽셀 값을 [0, 1] 범위로 클리핑 (모델 출력이 범위를 벗어날 경우 대비)
            predicted_image_np = np.clip(predicted_image_np, 0, 1)

            # --- 점수 계산 ---
            psnr_score = psnr(clean_image_np, predicted_image_np, data_range=1.0)
            
            # SSIM 계산 시 win_size는 이미지 크기보다 작아야 하며 홀수여야 함
            # (256, 256) 이미지이므로 7로 고정해도 안전
            ssim_score = ssim(clean_image_np, predicted_image_np, data_range=1.0, channel_axis=-1, win_size=7)

            total_psnr += psnr_score
            total_ssim += ssim_score
            image_count += 1
            
            print(f"Image {image_count}/{len(test_loader)} - PSNR: {psnr_score:.2f}, SSIM: {ssim_score:.4f}")

    # 5. 최종 평균 점수 계산 및 출력
    avg_psnr = total_psnr / image_count
    avg_ssim = total_ssim / image_count

    print("\n" + "="*30)
    print("     Evaluation Complete     ")
    print("="*30)
    print(f"Total Test Images: {image_count}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("="*30)
