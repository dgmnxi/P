# scripts/test_model_similarity.py

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import sys
import torch.nn.functional as F

# --- 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from app.model import resnet18_transfer_learning
    from app.separator import separate_audio, mel_spectrogram
except ImportError as e:
    print(f"Error: {e}")
    print("Could not import from 'app' module. Make sure you are running this script from the project root.")
    sys.exit(1)

# --- 사용자 설정: 여기에 비교하고 싶은 파일 쌍을 정의하세요 ---
TEST_PAIRS = {
    "positive_pairs": [
        ("positive_sample_1A.mp3", "positive_sample_1B.mp3"),
    ],
    "negative_pairs": [
        ("negative_sample_1.mp3", "negative_sample_2.mp3"),
    ]
}
# ---------------------------------------------------------

def get_chunk_embeddings(audio_path, model, device, segment_duration=5.0):
    """
    하나의 오디오 파일에서 모든 5초 구간의 임베딩 벡터 리스트를 추출합니다.
    """
    print(f"\n'{os.path.basename(audio_path)}' 파일 처리 중...")
    try:
        instrumental, sr = separate_audio(audio_path, device=device)
        if instrumental is None: return None
        
        if instrumental.dim() > 1: instrumental = instrumental.mean(dim=0)

        num_samples_per_segment = int(segment_duration * sr)
        segments = list(instrumental.split(num_samples_per_segment, dim=-1))
        if segments and segments[-1].shape[-1] < num_samples_per_segment:
            segments.pop()

        if not segments: return None

        batch_embeddings = []
        with torch.no_grad():
            for seg in tqdm(segments, desc="  - 임베딩 추출 중"):
                if seg.shape[-1] < num_samples_per_segment: continue
                seg = seg[..., :num_samples_per_segment]
                
                mel_spec = mel_spectrogram(seg.unsqueeze(0), sample_rate=sr).to(device)
                embedding = model(mel_spec).cpu()
                batch_embeddings.append(embedding)

        return torch.cat(batch_embeddings, dim=0) if batch_embeddings else None

    except Exception as e:
        print(f"  - 처리 중 오류 발생: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="학습된 모델의 구간별 유사도 측정 성능을 테스트합니다.")
    default_model_path = os.path.join(project_root, "models", "best_model.pth")
    default_test_dir = os.path.join(project_root, "data", "test")

    parser.add_argument('--model-path', type=str, default=default_model_path, help="학습된 모델 파일 경로")
    parser.add_argument('--test-dir', type=str, default=default_test_dir, help="테스트용 오디오 파일이 있는 디렉토리")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="사용할 장치")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다. 경로를 확인하세요: {args.model_path}")
        return

    model = resnet18_transfer_learning().to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    print(f"모델 로드 완료: {args.model_path}")

    song_chunk_embeddings = {}
    all_test_files = set(f for pair in TEST_PAIRS["positive_pairs"] for f in pair) | \
                   set(f for pair in TEST_PAIRS["negative_pairs"] for f in pair)

    for filename in all_test_files:
        file_path = os.path.join(args.test_dir, filename)
        if not os.path.exists(file_path):
            print(f"\n경고: '{filename}' 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue
        embeddings = get_chunk_embeddings(file_path, model, args.device)
        if embeddings is not None:
            song_chunk_embeddings[filename] = embeddings

    print("\n--- 구간별 유사도 측정 결과 ---")
    print("두 곡의 모든 구간을 교차 비교하여 가장 유사한 구간의 값을 찾습니다.")

    print("\n[ Positive Pairs (유사 구간이 존재해야 하는 쌍) ]")
    for song1_name, song2_name in TEST_PAIRS["positive_pairs"]:
        if song1_name in song_chunk_embeddings and song2_name in song_chunk_embeddings:
            chunks1 = song_chunk_embeddings[song1_name]
            chunks2 = song_chunk_embeddings[song2_name]
            
            # 모든 구간 쌍 간의 거리/유사도 계산
            distance_matrix = torch.cdist(chunks1, chunks2, p=2)
            similarity_matrix = F.cosine_similarity(chunks1.unsqueeze(1), chunks2.unsqueeze(0), dim=2)
            
            min_distance = torch.min(distance_matrix).item()
            max_similarity = torch.max(similarity_matrix).item()
            
            print(f"- '{song1_name}' vs '{song2_name}'")
            print(f"  - Minimum Distance (가장 가까운 구간 거리): {min_distance:.4f}")
            print(f"  - Maximum Similarity (가장 유사한 구간 유사도): {max_similarity:.4f}")

    print("\n[ Negative Pairs (유사 구간이 없어야 하는 쌍) ]")
    for song1_name, song2_name in TEST_PAIRS["negative_pairs"]:
        if song1_name in song_chunk_embeddings and song2_name in song_chunk_embeddings:
            chunks1 = song_chunk_embeddings[song1_name]
            chunks2 = song_chunk_embeddings[song2_name]

            distance_matrix = torch.cdist(chunks1, chunks2, p=2)
            similarity_matrix = F.cosine_similarity(chunks1.unsqueeze(1), chunks2.unsqueeze(0), dim=2)

            min_distance = torch.min(distance_matrix).item()
            max_similarity = torch.max(similarity_matrix).item()

            print(f"- '{song1_name}' vs '{song2_name}'")
            print(f"  - Minimum Distance: {min_distance:.4f}")
            print(f"  - Maximum Similarity: {max_similarity:.4f}")

if __name__ == '__main__':
    main()