# GPU를 사용하더라도 몇 시간에서 며칠이 걸릴 수 있는 파일
# 조심해서 실행할 것...
# 해당 파일을 통해 rrbc_model_trained.pth 획득

import torch
import torch.nn as nn
import torch.optim as optim # 옵티마이저
import torch.nn.functional as F # 손실 함수 등

# --- 1. RRBC 모델 ---
# --- 1-1. RRBC의 핵심 CNN 블록 정의 ---
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

# --- 1-2. ConvLSTM 셀 정의 ---
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

# --- 1-3. 최종 RRBC 전체 모델 조립 ---
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


# --- 2. 데이터 로더 ---
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image # 이미지 로딩을 위한 라이브러리
import os # 파일 경로 다루기

# --- 2-1. 데이터셋 사용 설명서 만들기 (Dataset 클래스) ---
class RainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): rainy와 clean 폴더가 있는 디렉토리 경로.
            transform (callable, optional): 샘플에 적용될 전처리(transform).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.rainy_dir = os.path.join(self.root_dir, 'rainy')
        self.clean_dir = os.path.join(self.root_dir, 'clean')
        
        # rainy 폴더에 있는 파일 목록 기준
        self.image_files = os.listdir(self.rainy_dir)

    def __len__(self):
        # 데이터셋의 총 이미지 개수를 반환
        return len(self.image_files)

    def __getitem__(self, idx):
        # idx번째 이미지를 불러오는 방법 정의
        img_name = self.image_files[idx]
        
        rainy_path = os.path.join(self.rainy_dir, img_name)
        clean_path = os.path.join(self.clean_dir, img_name)
        
        # PIL 라이브러리로 이미지 오픈
        rainy_image = Image.open(rainy_path).convert("RGB")
        clean_image = Image.open(clean_path).convert("RGB")
        
        # 전처리(transform)가 정의 시
        if self.transform:
            rainy_image = self.transform(rainy_image)
            clean_image = self.transform(clean_image)
            
        return rainy_image, clean_image

# --- 3. 훈련 루프 ---
if __name__ == '__main__':
    # --- 3-1. 모델 및 기본 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RRBC_Net(num_stages=3).to(device)

    # --- 3-2. 훈련에 필요한 요소 정의 ---
    # 옵티마이저 (모델의 파라미터를 업데이트하는 방법)
    # Adam이 일반적으로 성능이 좋아 많이 사용되서 선택
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 손실 함수 (모델의 예측이 정답과 얼마나 다른지 계산)
    # 이미지 복원에는 L1 손실(MAE)이나 L2 손실(MSE)이 주로 사용
    loss_function = nn.L1Loss() 

    # 데이터 로더 (대용량 데이터를 미니배치 단위로 공급)
    # --- 2-2. 이미지에 적용할 전처리 정의 ---
    # 이미지를 PyTorch 텐서로 변환. 픽셀 값을 0~1로 정규화
    transform = transforms.Compose([
        transforms.ToTensor() 
    ])

    # --- 2-3. 훈련용 및 검증용 데이터셋 생성 ---
    train_dataset = RainDataset(root_dir='data/train', transform=transform)
    val_dataset = RainDataset(root_dir='data/validation', transform=transform)
    
    # --- 2-4. 주방 보조(DataLoader) 고용! ---
    # BATCH_SIZE = 한 번에 모델에게 보여줄 이미지 개수. GPU 메모리에 따라 조절.
    BATCH_SIZE = 8 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3-3. 훈련 루프 시작 ---

    num_epochs = 10 # 전체 데이터셋을 총 10번 반복하여 학습

    for epoch in range(num_epochs):
        # ==================== 훈련 단계 ====================
        model.train() # 모델을 훈련 모드로 설정 
        print(f"Epoch {epoch+1}/{num_epochs} - Training...")

        # train_loader에서 미니배치를 하나씩 가져와 훈련
        for rainy_images, clean_images in train_loader:
            # 데이터를 device(GPU)로 이동
            rainy_images = rainy_images.to(device)
            clean_images = clean_images.to(device)

            # 1. 옵티마이저의 기울기 초기화
            optimizer.zero_grad()

            # 2. 모델 예측
            predicted_images = model(rainy_images)

            # 3. 손실 계산
            loss = loss_function(predicted_images, clean_images)

            # 4. 역전파 (기울기 계산)
            loss.backward()

            # 5. 옵티마이저로 모델 파라미터 업데이트
            optimizer.step()
        
        print("Training finished for this epoch.")

        model.eval()
        print("Validating...")
            
        with torch.no_grad():
            total_val_loss = 0
            # 검증 로더를 사용하여 성능 평가
            # 훈련 단계와 유사하게 손실이나 PSNR 같은 성능 지표 계산
            for rainy_images, clean_images in val_loader:
                rainy_images = rainy_images.to(device)
                clean_images = clean_images.to(device)
                predicted_images = model(rainy_images)
                val_loss = loss_function(predicted_images, clean_images)
                total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            print("Validation finished for this epoch.")
            print("-" * 30)

# 훈련 완료 후 모델 가중치 저장
torch.save(model.state_dict(), 'rrbc_model_trained.pth')
print("Model training complete and saved.")
