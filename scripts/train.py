'''
트립렛 손실(Triplet Loss)을 사용하여 모델을 학습하는 스크립트입니다.
data/processed 디렉토리에서 전처리된 멜 스펙토그램 데이터를 로드하고,
앵커(anchor), 포지티브(positive), 네거티브(negative) 샘플을 생성하여 모델을 학습합니다.
학습된 모델은 지정된 디렉토리에 저장됩니다.
A: 현재 학습할려는 노래의 악기
P: 같은 노래의 같은 악기 / 다른 구간
N: 0.5의 확률로 1. 다른 노래의 같은 악기 2. 같은 악기, 다른 노래 (hard negative -> 같은 기타여도 통기타, 일렉기타 등을 구분할 수 있게 함)


'''



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
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- 경로 설정 ---
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
    def __init__(self, data_dir, song_list, instruments, data_map):
        self.data_dir = data_dir
        self.song_list = song_list
        self.instruments = instruments
        self.data_map = data_map
        
        self.samples = []
        for song in self.song_list:
            for instrument in self.instruments:
                if song in self.data_map[instrument]:
                    self.samples.extend(self.data_map[instrument][song])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor_path = self.samples[index]
        anchor_parts = anchor_path.split(os.sep)
        anchor_instrument = anchor_parts[-2]
        anchor_song = '_'.join(anchor_parts[-1].split('_')[:-1])

        positive_list = self.data_map[anchor_instrument][anchor_song]
        positive_path = random.choice([p for p in positive_list if p != anchor_path]) if len(positive_list) > 1 else anchor_path

        if random.random() < 0.5:
            neg_inst = random.choice([i for i in self.instruments if i != anchor_instrument])
            neg_song = random.choice(list(self.data_map[neg_inst].keys()))
        else:
            neg_inst = anchor_instrument
            other_songs = [s for s in self.data_map[neg_inst].keys() if s != anchor_song and s in self.song_list]
            if not other_songs:
                neg_inst = random.choice([i for i in self.instruments if i != anchor_instrument])
                neg_song = random.choice(list(self.data_map[neg_inst].keys()))
            else:
                neg_song = random.choice(other_songs)

        negative_path = random.choice(self.data_map[neg_inst][neg_song])

        anchor = torch.load(anchor_path)
        positive = torch.load(positive_path)
        negative = torch.load(negative_path)

        return anchor, positive, negative

def main():
    parser = argparse.ArgumentParser(description="Triplet Loss로 모델을 학습합니다.")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_data_dir = os.path.join(project_root, "data/processed")
    default_model_dir = os.path.join(project_root, "models")

    parser.add_argument('--data-dir', type=str, default=default_data_dir, help="전처리된 데이터가 있는 디렉토리")
    parser.add_argument('--model-dir', type=str, default=default_model_dir, help="학습된 모델을 저장할 디렉토리")
    parser.add_argument('--epochs', type=int, default=50, help="학습할 에포크 수")
    parser.add_argument('--batch-size', type=int, default=32, help="배치 사이즈")
    parser.add_argument('--lr', type=float, default=1e-4, help="학습률")
    parser.add_argument('--validation-split', type=float, default=0.1, help="검증 데이터셋 비율")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="학습에 사용할 장치")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    # --- 1. 전체 데이터 정보 수집 ---
    logging.info("전체 데이터셋 정보를 구성합니다...")
    all_files = glob(os.path.join(args.data_dir, '**', '*.pt'), recursive=True)
    if not all_files:
        raise ValueError(f"'{args.data_dir}'에서 .pt 파일을 찾을 수 없습니다.")

    data_map = defaultdict(lambda: defaultdict(list))
    all_songs = set()
    for f in all_files:
        parts = f.split(os.sep)
        instrument = parts[-2]
        song_name = '_'.join(parts[-1].split('_')[:-1])
        data_map[instrument][song_name].append(f)
        all_songs.add(song_name)
    
    instruments = list(data_map.keys())
    all_songs = sorted(list(all_songs))
    random.shuffle(all_songs)

    # --- 2. 훈련/검증용으로 곡 목록 분리 ---
    split_idx = int(len(all_songs) * (1 - args.validation_split))
    train_songs = all_songs[:split_idx]
    val_songs = all_songs[split_idx:]

    logging.info(f"총 {len(all_songs)}곡 중 {len(train_songs)}곡은 훈련용, {len(val_songs)}곡은 검증용으로 분리")

    # --- 3. 데이터셋 및 데이터로더 생성 ---
    train_dataset = TripletSpectrogramDataset(args.data_dir, train_songs, instruments, data_map)
    val_dataset = TripletSpectrogramDataset(args.data_dir, val_songs, instruments, data_map)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 모델, 손실 함수, 옵티마이저
    model = resnet18_transfer_learning().to(args.device)
    loss_fn = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')

    logging.info("학습을 시작합니다...")
    for epoch in range(args.epochs):
        # --- 훈련 단계 ---
        model.train()
        total_train_loss = 0
        for anchor, positive, negative in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [훈련]"):
            anchor, positive, negative = anchor.to(args.device), positive.to(args.device), negative.to(args.device)
            optimizer.zero_grad()
            anchor_out, positive_out, negative_out = model(anchor), model(positive), model(negative)
            loss = loss_fn(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_dataloader)

        # --- 검증 단계 ---
        model.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        with torch.no_grad():
            for anchor, positive, negative in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [검증]"):
                anchor, positive, negative = anchor.to(args.device), positive.to(args.device), negative.to(args.device)
                anchor_out, positive_out, negative_out = model(anchor), model(positive), model(negative)
                
                loss = loss_fn(anchor_out, positive_out, negative_out)
                total_val_loss += loss.item()

                dist_pos = torch.pairwise_distance(anchor_out, positive_out)
                dist_neg = torch.pairwise_distance(anchor_out, negative_out)
                total_val_accuracy += (dist_pos < dist_neg).sum().item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = total_val_accuracy / len(val_dataset)

        logging.info(f"Epoch {epoch+1} 완료 | 훈련 Loss: {avg_train_loss:.4f} | 검증 Loss: {avg_val_loss:.4f} | 검증 정확도: {val_accuracy:.4f}")

        scheduler.step(avg_val_loss)

        # --- 최고 성능 모델 저장 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.model_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            logging.info(f"최고 검증 Loss / 모델 저장: {save_path}")

if __name__ == '__main__':
    main()