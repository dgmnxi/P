# scripts/run_search_test.py

import argparse
import os
import torch
import sys

# --- 경로 설정 (수정된 부분) ---
# 스크립트 파일의 위치를 기준으로, 한 단계 상위 폴더인 프로젝트 루트 경로를 계산합니다.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 'app' 모듈을 찾을 수 있도록 프로젝트 루트를 sys.path에 추가합니다.
if project_root not in sys.path:
    sys.path.append(project_root)
# --- 경로 설정 끝 ---

from app.model import resnet18_transfer_learning
from app.separator import separate_audio, mel_spectrogram
from app.search import VectorSearchEngine

# --- 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_WIDTH = 431  # 훈련 시 사용했던 5초 분량의 스펙트로그램 너비

def main():
    parser = argparse.ArgumentParser(description="특정 오디오 클립으로 유사도 검색을 테스트합니다.")
    parser.add_argument("input_file", type=str, help="검색할 원본 오디오 파일 경로")
    parser.add_argument("start_sec", type=float, help="검색할 구간의 시작 시간 (초)")
    parser.add_argument("end_sec", type=float, help="검색할 구간의 종료 시간 (초)")
    parser.add_argument("--instrument", type=str, default="other", help="분리할 악기 이름 (bass, drums, other, vocals)")
    parser.add_argument("--top_k", type=int, default=5, help="검색할 상위 결과 개수")
    args = parser.parse_args()

    # --- 1. 리소스 로드 ---
    print("--- 리소스 로딩 시작 ---")
    # 모델 로드
    model_path = os.path.join(project_root, "models/best_model.pth")
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    model = resnet18_transfer_learning().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"모델 로드 완료: {model_path}")

    # 검색 엔진 로드
    index_path = os.path.join(project_root, "indexes/timbre.index")
    metadata_path = os.path.join(project_root, "indexes/metadata.json")
    if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
        print(f"오류: 인덱스 또는 메타데이터 파일을 찾을 수 없습니다.")
        return
    search_engine = VectorSearchEngine(index_path, metadata_path)
    print("검색 엔진 로드 완료")
    print("--- 리소스 로딩 완료 ---\n")

    # --- 2. 쿼리 오디오 처리 ---
    print(f"--- 쿼리 처리 시작: '{os.path.basename(args.input_file)}' ({args.start_sec:.2f}s ~ {args.end_sec:.2f}s) ---")
    try:
        # 2-1. 음원 분리
        stems, sr = separate_audio(args.input_file, device=DEVICE)
        if stems is None or args.instrument not in stems:
            print(f"오류: '{args.instrument}' 트랙을 분리할 수 없습니다.")
            return
        
        instrumental_track = stems[args.instrument]
        print(f"[DEBUG] separate_audio 결과 (instrumental_track): {instrumental_track.shape}")

        # 2-2. 사용자가 지정한 구간만 잘라내기
        start_sample = int(args.start_sec * sr)
        end_sample = int(args.end_sec * sr)
        audio_clip = instrumental_track[:, start_sample:end_sample]
        print(f"[DEBUG] 구간 자르기 결과 (audio_clip): {audio_clip.shape}")

        if audio_clip.shape[1] == 0:
            print("오류: 지정된 시간대에 오디오 데이터가 없습니다.")
            return

        # --- 최종 수정: 스펙트로그램 생성 직전에 모노로 변환 ---
        # separate_audio는 스테레오([2, n])를 반환하므로, 여기서 모노로 변환합니다.
        mono_audio_clip = torch.mean(audio_clip, dim=0, keepdim=True)
        print(f"[DEBUG] 모노 변환 결과 (mono_audio_clip): {mono_audio_clip.shape}")
        # --- 수정 끝 ---

        # 2-3. 스펙트로그램 변환
        long_spectrogram = mel_spectrogram(mono_audio_clip, sample_rate=sr)
        print(f"[DEBUG] mel_spectrogram 결과 (long_spectrogram): {long_spectrogram.shape}")

        # 2-4. 슬라이딩 윈도우로 쿼리 벡터 생성 (main.py 로직과 동일)
        total_frames = long_spectrogram.shape[-1]
        window_size = TARGET_WIDTH
        stride = window_size // 2

        chunks = []
        if total_frames <= window_size:
            padded_chunk = torch.zeros(1, long_spectrogram.shape[1], window_size, device=DEVICE)
            padded_chunk[..., :total_frames] = long_spectrogram
            chunks.append(padded_chunk)
        else:
            for i in range(0, total_frames - window_size + 1, stride):
                chunk = long_spectrogram[..., i:i + window_size]
                chunks.append(chunk)
        
        if not chunks:
            print("오류: 유효한 오디오 조각을 추출할 수 없습니다.")
            return

        batch = torch.cat(chunks, dim=0)
        
        # [배치, 높이, 너비] 형태를 -> [배치, 채널=1, 높이, 너비] 형태로 변환
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)

        batch = batch.to(DEVICE)

        print(f"[DEBUG] 모델 입력 직전 (batch): {batch.shape}")
        with torch.no_grad():
            all_embeddings = model(batch)
        query_vector = torch.mean(all_embeddings, dim=0, keepdim=True)
        print("쿼리 벡터 생성 완료")

    except Exception as e:
        print(f"쿼리 처리 중 오류 발생: {e}")
        return

    # --- 3. 유사도 검색 및 결과 출력 ---
    print("\n--- 유사도 검색 결과 ---")
    results = search_engine.search(query_vector.cpu().numpy(), top_k=args.top_k)

    if not results:
        print("유사한 곡을 찾지 못했습니다.")
    else:
        for res in results:
            print(f"  - ID: {res['id']}, Similarity: {res['similarity']:.4f}")
            print(f"    Song: {res['song_name']}, Instrument: {res['instrument']}")
            print(f"    Time: {res['start_sec']:.2f}s ~ {res['end_sec']:.2f}s\n")

if __name__ == "__main__":
    main()