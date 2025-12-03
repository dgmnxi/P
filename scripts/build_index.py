'''
모델을 사용하여 데이터의 임베딩을 추출하고 Faiss 인덱스를 구축합니다.
- IndexFlatIP를 사용하여 코사인 유사도 기반 검색을 지원합니다.
- 모든 임베딩 벡터는 L2 정규화를 거쳐 단위 벡터로 만들어 인덱스에 추가합니다.
'''
import os
import sys
import torch
import numpy as np
import faiss
from glob import glob
from tqdm import tqdm
import json
import argparse
from collections import defaultdict

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

# --- 상수 정의 ---
EMBEDDING_DIM = 512  # ResNet18 전이 학습 모델의 출력 차원

def build_index(data_dir, model_path, index_dir, device):
    """
    Faiss 인덱스와 메타데이터를 생성합니다.
    """
    os.makedirs(index_dir, exist_ok=True)

    # --- 1. 모델 로드 ---
    print("--- 1. 모델 로딩 시작 ---")
    model = resnet18_transfer_learning()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("--- 모델 로딩 완료 ---\n")

    # --- 2. 데이터 경로 수집 ---
    print("--- 2. 데이터 경로 수집 시작 ---")
    all_files = glob(os.path.join(data_dir, '**', '*.pt'), recursive=True)
    if not all_files:
        raise ValueError(f"'{data_dir}'에서 .pt 파일을 찾을 수 없습니다.")
    print(f"총 {len(all_files)}개의 .pt 파일 발견")
    print("--- 데이터 경로 수집 완료 ---\n")

    # --- 3. 임베딩 추출 ---
    print("--- 3. 임베딩 추출 시작 ---")
    embeddings = []
    metadata = {}
    with torch.no_grad():
        for i, file_path in enumerate(tqdm(all_files, desc="임베딩 추출 중")):
            tensor = torch.load(file_path).to(device)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            
            embedding = model(tensor).cpu().numpy()
            embeddings.append(embedding)

            # 메타데이터 생성
            parts = file_path.split(os.sep)
            instrument = parts[-2]
            filename = parts[-1]
            song_name = '_'.join(filename.split('_')[:-1])
            chunk_index = filename.split('_')[-1].replace('.pt', '')
            
            # start_sec, end_sec 계산 (prepare_data.py 로직과 동일하게)
            segment_duration = 5 # 5초
            start_sec = int(chunk_index) * segment_duration
            end_sec = start_sec + segment_duration

            metadata[str(i)] = {
                "song_name": song_name,
                "instrument": instrument,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "original_path": file_path
            }

    embeddings = np.vstack(embeddings)
    print("--- 임베딩 추출 완료 ---\n")

    # --- 4. Faiss 인덱스 구축 (IndexFlatIP) ---
    print("--- 4. Faiss 인덱스 구축 시작 ---")
    
    # L2 정규화 (코사인 유사도 계산을 위해)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    
    print(f"인덱스 구축 완료. 총 {index.ntotal}개의 벡터가 추가되었습니다.")
    
    # 인덱스 및 메타데이터 저장
    index_path = os.path.join(index_dir, "timbre.index")
    metadata_path = os.path.join(index_dir, "metadata.json")
    
    faiss.write_index(index, index_path)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
        
    print(f"인덱스가 '{index_path}'에 저장되었습니다.")
    print(f"메타데이터가 '{metadata_path}'에 저장되었습니다.")
    print("--- Faiss 인덱스 구축 완료 ---\n")

def main():
    parser = argparse.ArgumentParser(description="모델과 데이터를 사용하여 Faiss 인덱스를 구축합니다.")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    default_data_dir = os.path.join(project_root, "data/processed")
    default_model_path = os.path.join(project_root, "models/best_model.pth")
    default_index_dir = os.path.join(project_root, "indexes")

    parser.add_argument('--data-dir', type=str, default=default_data_dir, help="전처리된 데이터(.pt)가 있는 디렉토리")
    parser.add_argument('--model-path', type=str, default=default_model_path, help="학습된 모델 파일(.pth) 경로")
    parser.add_argument('--index-dir', type=str, default=default_index_dir, help="생성된 인덱스와 메타데이터를 저장할 디렉토리")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="임베딩 추출에 사용할 장치")
    
    args = parser.parse_args()

    print("인덱스 빌드 스크립트를 시작합니다.")
    print(f" - 데이터 디렉토리: {args.data_dir}")
    print(f" - 모델 경로: {args.model_path}")
    print(f" - 인덱스 저장 디렉토리: {args.index_dir}")
    print(f" - 사용 장치: {args.device}\n")

    build_index(args.data_dir, args.model_path, args.index_dir, args.device)

    print("모든 작업이 완료되었습니다.")

if __name__ == '__main__':
    main()
