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
                    # 각 샘플에 (파일 경로, 곡 ID, 악기 ID)를 튜플로 저장
                    song_id = self.song_list.index(song)
                    instrument_id = self.instruments.index(instrument)
                    for path in self.data_map[instrument][song]:
                        self.samples.append((path, song_id, instrument_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor_path, anchor_song_id, anchor_instrument_id = self.samples[index]
        
        anchor_song_name = self.song_list[anchor_song_id]
        anchor_instrument_name = self.instruments[anchor_instrument_id]

        # Positive 샘플 선택
        positive_list = self.data_map[anchor_instrument_name][anchor_song_name]
        positive_path = random.choice([p for p in positive_list if p != anchor_path]) if len(positive_list) > 1 else anchor_path

        anchor = torch.load(anchor_path)
        positive = torch.load(positive_path)

        # 곡 ID와 악기 ID를 함께 반환
        return anchor, positive, torch.tensor(anchor_song_id), torch.tensor(anchor_instrument_id)

def main():
    parser = argparse.ArgumentParser(description="Triplet Loss로 모델을 학습합니다.")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_data_dir = os.path.join(project_root, "data/processed")
    default_model_dir = os.path.join(project_root, "models")

    parser.add_argument('--data-dir', type=str, default=default_data_dir, help="전처리된 데이터가 있는 디렉토리")
    parser.add_argument('--model-dir', type=str, default=default_model_dir, help="학습된 모델을 저장할 디렉토리")
    parser.add_argument('--epochs', type=int, default=50, help="학습할 에포크 수")
    parser.add_argument('--batch-size', type=int, default=32, help="배치 사이즈")
    parser.add_argument('--lr', type=float, default=1e-5, help="학습률")
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
    # `all_songs`와 `instruments`를 id로 사용하므로 순서가 중요
    train_dataset = TripletSpectrogramDataset(args.data_dir, all_songs, instruments, data_map)
    # 검증 데이터셋은 기존 방식을 유지하거나, 단순 분류 정확도로 변경할 수 있습니다.
    # 여기서는 간단하게 val_dataset도 동일한 클래스를 사용합니다.
    val_dataset = TripletSpectrogramDataset(args.data_dir, all_songs, instruments, data_map)
    
    # train_sampler와 val_sampler를 만들어 데이터를 분리합니다.
    train_indices = [i for i, s in enumerate(train_dataset.samples) if train_dataset.song_list[s[1]] in train_songs]
    val_indices = [i for i, s in enumerate(val_dataset.samples) if val_dataset.song_list[s[1]] in val_songs]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    # 모델, 손실 함수, 옵티마이저
    model = resnet18_transfer_learning().to(args.device)
    # `reduction='none'`으로 설정하여 각 triplet에 대한 loss를 개별적으로 계산
    loss_fn = nn.TripletMarginLoss(margin=1.0, reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')

    logging.info("학습을 시작합니다...")
    for epoch in range(args.epochs):
        # --- 훈련 단계 (온라인 하드 네거티브 마이닝) ---
        model.train()
        total_train_loss = 0
        train_triplets_found = 0
        for anchor, positive, song_ids, _ in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [훈련]"):
            anchor, positive = anchor.to(args.device), positive.to(args.device)
            
            # (anchor, positive) 쌍의 임베딩 계산
            anchor_out = model(anchor)
            positive_out = model(positive)
            
            # 모든 임베딩을 하나로 모음
            all_embeddings = torch.cat((anchor_out, positive_out), dim=0)
            all_song_ids = torch.cat((song_ids, song_ids), dim=0)

            # 거리 행렬 계산
            dist_matrix = torch.cdist(anchor_out, all_embeddings)

            # 하드 네거티브 찾기
            hard_negatives = []
            for i in range(anchor_out.size(0)):
                anchor_song_id = song_ids[i]
                
                # 네거티브 후보: 앵커와 다른 곡 ID를 가진 모든 샘플
                is_negative = all_song_ids != anchor_song_id
                
                # 앵커와 네거티브 후보들 간의 거리
                neg_dists = dist_matrix[i][is_negative]
                
                if len(neg_dists) == 0: continue

                # 하드 네거티브: 앵커와 가장 가까운 네거티브
                hard_negative_idx = torch.argmin(neg_dists)
                hard_negatives.append(all_embeddings[is_negative][hard_negative_idx])

            if not hard_negatives:
                continue
            
            negative_out = torch.stack(hard_negatives)
            
            # Triplet Loss 계산
            loss = loss_fn(anchor_out[:negative_out.size(0)], positive_out[:negative_out.size(0)], negative_out)
            
            # 유효한 triplet (loss > 0)만으로 평균 loss 계산
            valid_triplets = loss > 0
            num_valid_triplets = valid_triplets.sum().item()

            if num_valid_triplets > 0:
                mean_loss = loss[valid_triplets].mean()
                
                optimizer.zero_grad()
                mean_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping 추가
                optimizer.step()
                
                total_train_loss += mean_loss.item() * num_valid_triplets
                train_triplets_found += num_valid_triplets

        avg_train_loss = total_train_loss / train_triplets_found if train_triplets_found > 0 else 0

        # --- 검증 단계 ---
        model.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        val_triplets_found = 0
        with torch.no_grad():
            for anchor, positive, song_ids, _ in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [검증]"):
                anchor, positive = anchor.to(args.device), positive.to(args.device)
                anchor_out = model(anchor)
                positive_out = model(positive)

                all_embeddings = torch.cat((anchor_out, positive_out), dim=0)
                all_song_ids = torch.cat((song_ids, song_ids), dim=0)
                dist_matrix = torch.cdist(anchor_out, all_embeddings)

                hard_negatives = []
                for i in range(anchor_out.size(0)):
                    anchor_song_id = song_ids[i]
                    is_negative = all_song_ids != anchor_song_id
                    neg_dists = dist_matrix[i][is_negative]
                    if len(neg_dists) == 0: continue
                    hard_negative_idx = torch.argmin(neg_dists)
                    hard_negatives.append(all_embeddings[is_negative][hard_negative_idx])

                if not hard_negatives: continue
                
                negative_out = torch.stack(hard_negatives)
                
                loss = loss_fn(anchor_out[:negative_out.size(0)], positive_out[:negative_out.size(0)], negative_out)
                valid_triplets = loss > 0
                num_valid_triplets = valid_triplets.sum().item()

                if num_valid_triplets > 0:
                    total_val_loss += loss[valid_triplets].sum().item()
                    val_triplets_found += num_valid_triplets

                # 정확도 계산 (앵커-포지티브 거리가 앵커-네거티브 거리보다 작은 경우)
                dist_pos = torch.pairwise_distance(anchor_out[:negative_out.size(0)], positive_out[:negative_out.size(0)])
                dist_neg = torch.pairwise_distance(anchor_out[:negative_out.size(0)], negative_out)
                total_val_accuracy += (dist_pos < dist_neg).sum().item()

        avg_val_loss = total_val_loss / val_triplets_found if val_triplets_found > 0 else 0
        # 정확도는 전체 검증 샘플 중 성공한 비율로 계산
        val_accuracy = total_val_accuracy / len(val_indices) if len(val_indices) > 0 else 0

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