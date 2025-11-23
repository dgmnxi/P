import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
import random
import logging
from tqdm import tqdm
import sys
from collections import defaultdict

# 프로젝트 루트를 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from app.model import resnet18_transfer_learning
except ImportError as e:
    print(f"Error: {e}")
    print("Could not import 'resnet18_transfer_learning'. Make sure you are running this script from the project root.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TripletSpectrogramDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.data_map = defaultdict(lambda: defaultdict(list))
        self.instruments = []

        logging.info("데이터셋을 구성합니다...")
        # data/processed/{instrument}/{song_name}_{clip_id}.pt
        all_files = glob(os.path.join(data_dir, '**', '*.pt'), recursive=True)
        
        for f in all_files:
            parts = f.split(os.sep)
            instrument = parts[-2]
            filename = parts[-1]
            song_name = '_'.join(filename.split('_')[:-1])
            
            self.data_map[instrument][song_name].append(f)
            self.samples.append(f)

        self.instruments = list(self.data_map.keys())
        if not self.samples:
            raise ValueError(f"'{data_dir}'에서 데이터를 찾을 수 없습니다. prepare_data.py를 먼저 실행하세요.")
        logging.info(f"총 {len(self.samples)}개의 샘플과 {len(self.instruments)}개의 악기를 찾았습니다.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 1. Anchor 샘플 선택
        anchor_path = self.samples[index]
        anchor_parts = anchor_path.split(os.sep)
        anchor_instrument = anchor_parts[-2]
        anchor_song = '_'.join(anchor_parts[-1].split('_')[:-1])

        # 2. Positive 샘플 선택 (같은 노래, 같은 악기, 다른 클립)
        positive_list = self.data_map[anchor_instrument][anchor_song]
        positive_path = random.choice([p for p in positive_list if p != anchor_path]) if len(positive_list) > 1 else anchor_path

        # 3. Negative 샘플 선택 (다른 노래, 다른 악기)
        negative_instrument = random.choice([i for i in self.instruments if i != anchor_instrument])
        negative_song = random.choice(list(self.data_map[negative_instrument].keys()))
        negative_path = random.choice(self.data_map[negative_instrument][negative_song])

        anchor = torch.load(anchor_path)
        positive = torch.load(positive_path)
        negative = torch.load(negative_path)

        return anchor, positive, negative

def main():
    parser = argparse.ArgumentParser(description="Triplet Loss로 모델을 학습합니다.")
    parser.add_argument('--data-dir', type=str, default="data/processed", help="전처리된 데이터가 있는 디렉토리")
    parser.add_argument('--model-dir', type=str, default="models", help="학습된 모델을 저장할 디렉토리")
    parser.add_argument('--epochs', type=int, default=20, help="학습할 에포크 수")
    parser.add_argument('--batch-size', type=int, default=16, help="배치 사이즈")
    parser.add_argument('--lr', type=float, default=1e-4, help="학습률 (Learning Rate)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="학습에 사용할 장치")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    
    # 데이터셋 및 데이터로더
    dataset = TripletSpectrogramDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 모델, 손실 함수, 옵티마이저
    model = resnet18_transfer_learning().to(args.device)
    loss_fn = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logging.info("학습을 시작합니다...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for anchor, positive, negative in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            anchor, positive, negative = anchor.to(args.device), positive.to(args.device), negative.to(args.device)
            
            optimizer.zero_grad()
            
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            
            loss = loss_fn(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1} 완료. 평균 Loss: {avg_loss:.4f}")
        
        # 모델 저장
        save_path = os.path.join(args.model_dir, f"resnet18_epoch_{epoch+1:02d}.pth")
        torch.save(model.state_dict(), save_path)
        logging.info(f"모델 저장 완료: {save_path}")

if __name__ == '__main__':
    main()