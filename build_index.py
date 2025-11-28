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

import os
import torch
import faiss
import json
import numpy as np
from glob import glob
from tqdm import tqdm

# --- 수정된 부분: model.py에서 함수를 직접 임포트 ---
from app.model import resnet18_transfer_learning
from app.separator import separate_audio, mel_spectrogram

# --- 설정 ---
RAW_DATA_DIR = "data/raw"
MODEL_PATH = "models/audio_resnet_best.pth"
INDEX_DIR = "indexes"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EMBEDDING_DIM = 128

def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    print(f"Loading Model from {MODEL_PATH}...")
    # --- 수정된 부분: 클래스 생성 대신 함수 호출 ---
    model = resnet18_transfer_learning(output_dim=EMBEDDING_DIM, freeze_features=False).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"Warning: Model weights not found at {MODEL_PATH}. Using initial transfer learning weights for indexing.")
    model.eval()

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata = {}
    current_id = 0

    audio_files = glob(os.path.join(RAW_DATA_DIR, "*.*"))
    audio_files = [f for f in audio_files if f.lower().endswith(('.mp3', '.wav', '.flac'))]
    
    print(f"Found {len(audio_files)} songs. Starting indexing process...")

    for audio_path in tqdm(audio_files, desc="Processing Songs"):
        song_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        try:
            stems, sr = separate_audio(audio_path, device=DEVICE)
            
            for instrument, waveform in stems.items():
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=0)

                samples_per_segment = int(5.0 * sr)
                total_samples = waveform.shape[-1]
                
                batch_specs = []
                batch_meta = []

                for start_idx in range(0, total_samples, samples_per_segment):
                    end_idx = start_idx + samples_per_segment
                    if end_idx > total_samples: break

                    segment = waveform[start_idx:end_idx]
                    spec = mel_spectrogram(segment.unsqueeze(0), sample_rate=sr)
                    batch_specs.append(spec)
                    
                    batch_meta.append({
                        "song_name": song_name,
                        "instrument": instrument,
                        "start_sec": start_idx / sr,
                        "end_sec": end_idx / sr
                    })

                    # 배치 사이즈가 차면 처리
                    if len(batch_specs) >= BATCH_SIZE:
                        batch_tensor = torch.cat(batch_specs).to(DEVICE)
                        with torch.no_grad():
                            embeddings = model(batch_tensor).cpu().numpy()
                        
                        index.add(embeddings)
                        for i in range(len(embeddings)):
                            metadata[str(current_id)] = batch_meta[i]
                            current_id += 1
                        
                        batch_specs, batch_meta = [], []

                # 남은 배치 처리
                if batch_specs:
                    batch_tensor = torch.cat(batch_specs).to(DEVICE)
                    with torch.no_grad():
                        embeddings = model(batch_tensor).cpu().numpy()
                    
                    index.add(embeddings)
                    for i in range(len(embeddings)):
                        metadata[str(current_id)] = batch_meta[i]
                        current_id += 1

        except Exception as e:
            print(f"Error processing {song_name}: {e}")
            continue

    index_path = os.path.join(INDEX_DIR, "timbre.index")
    meta_path = os.path.join(INDEX_DIR, "metadata.json")
    
    print(f"Saving Index to {index_path}...")
    faiss.write_index(index, index_path)
    
    print(f"Saving Metadata to {meta_path}...")
    with open(meta_path, "w", encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("--- Indexing Complete! ---")

if __name__ == "__main__":
    build_index()