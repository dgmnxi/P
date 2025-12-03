'''
data/raw에 있는 오디오 파일들을 
악기 분리 / 5초 분할 / Mel-Spectrogram 변환 / AudioResNet 임베딩 추출
단계를 거쳐 Faiss 인덱스를 구축하는 스크립트입니다.

출력: indexes/timbre.index (Faiss 인덱스 파일)
      indexes/metadata.json (메타데이터 매핑 파일)
      metadata.json 예시:
      {
            "0": {
                "song_name": "song1",
                "instrument": "vocals",
                "start_sec": 0.0,
                "end_sec": 5.0
            },
            "1": {
                "song_name": "song1",
                "instrument": "vocals",
                "start_sec": 5.0,
                "end_sec": 10.0
            },
'''

# build_index.py

import os
import torch
import faiss
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import sys

# --- 경로 설정 ---
project_root = os.path.dirname(os.path.abspath(__file__))
# 'app' 모듈을 찾기 위해 프로젝트 루트를 sys.path에 추가합니다.
if project_root not in sys.path:
    sys.path.append(project_root)

from app.model import resnet18_transfer_learning

# --- 설정 ---
# '..'를 모두 제거하여 모든 경로가 프로젝트 폴더 내에서 결정되도록 합니다.
PROCESSED_DATA_DIR = os.path.join(project_root, "data/processed")
MODEL_PATH = os.path.join(project_root, "models/best_model.pth")
INDEX_DIR = os.path.join(project_root, "indexes")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EMBEDDING_DIM = 128

def build_index_from_processed():
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    # 1. 모델 로드
    print(f"Loading Model from {MODEL_PATH}...")
    model = resnet18_transfer_learning().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL: Model weights not found at {MODEL_PATH}. Please run 'scripts/train.py' first.")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. 처리된 .pt 파일 목록 가져오기
    pt_files = glob(os.path.join(PROCESSED_DATA_DIR, '**', '*.pt'), recursive=True)
    if not pt_files:
        print(f"FATAL: No .pt files found in '{PROCESSED_DATA_DIR}'. Please run 'scripts/prepare_data.py' first.")
        return
    
    print(f"Found {len(pt_files)} processed chunks. Starting indexing process...")

    # 3. 인덱스와 메타데이터 초기화
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata = {}
    current_id = 0
    
    batch_specs = []
    batch_files_info = []

    with torch.no_grad():
        for pt_path in tqdm(pt_files, desc="Indexing Chunks"):
            try:
                # 파일 경로에서 정보 파싱
                parts = pt_path.replace('\\', '/').split('/')
                instrument = parts[-2]
                filename = parts[-1]
                song_name = '_'.join(filename.split('_')[:-1])
                chunk_id = int(filename.split('_')[-1].split('.')[0])

                # 스펙트로그램 텐서 로드
                spec = torch.load(pt_path, map_location=DEVICE)
                
                # --- ★★★★★ 최종 수정 ★★★★★ ---
                # 모든 텐서가 (1, 128, 431) 모양이라고 가정하고,
                # 배치에 추가하기 전에 unsqueeze(0)를 호출하여 (1, 1, 128, 431)로 만듭니다.
                if spec.shape == (1, 128, 431):
                    spec = spec.unsqueeze(0) # (1, 1, 128, 431)로 만듦
                else:
                    tqdm.write(f"Warning: Skipping {pt_path} due to unexpected shape {spec.shape}")
                    continue
                
                batch_specs.append(spec)
                
                # 메타데이터 정보 저장
                batch_files_info.append({
                    "song_name": song_name,
                    "instrument": instrument,
                    "start_sec": round(chunk_id * 5.0, 2),
                    "end_sec": round((chunk_id + 1) * 5.0, 2)
                })

                # 배치가 차면 처리
                if len(batch_specs) >= BATCH_SIZE:
                    # (1, 1, 128, 431) 모양의 텐서 리스트를 (BATCH_SIZE, 1, 128, 431) 모양으로 합침
                    batch_tensor = torch.cat(batch_specs, dim=0)
                    embeddings = model(batch_tensor).cpu().numpy()
                    
                    index.add(embeddings)
                    for info in batch_files_info:
                        metadata[str(current_id)] = info
                        current_id += 1
                    
                    batch_specs, batch_files_info = [], []

            except Exception as e:
                print(f"Error processing {pt_path}: {e}")
                continue

        # 남은 배치 처리
        if batch_specs:
            batch_tensor = torch.cat(batch_specs, dim=0)
            embeddings = model(batch_tensor).cpu().numpy()
            
            index.add(embeddings)
            for info in batch_files_info:
                metadata[str(current_id)] = info
                current_id += 1

    # 4. 최종 파일 저장
    index_path = os.path.join(INDEX_DIR, "timbre.index")
    meta_path = os.path.join(INDEX_DIR, "metadata.json")
    
    print(f"\nSaving Index with {index.ntotal} vectors to {index_path}...")
    faiss.write_index(index, index_path)
    
    print(f"Saving Metadata to {meta_path}...")
    with open(meta_path, "w", encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("--- Indexing Complete! ---")

if __name__ == "__main__":
    build_index_from_processed()