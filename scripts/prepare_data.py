'''
모델 훈련에 필요한 데이터 셋을 준비하는 스크립트 입니다.
원본 오디오 파일을 로드하고, Demucs를 사용하여 악기별로 분리한 후,
각 악기별로 지정된 길이로 분할하고 Mel 스펙토그램으로 변환하여 저장합니다.
이 때, 각 악기별로 5초 단위로 분할하여 저장하며 , 저장 형식은 PyTorch 텐서(.pt / 1X128x413) 입니다.

'''


import argparse
import os
import torch
import torchaudio
from glob import glob
import logging
from tqdm import tqdm
import sys

# 프로젝트 루트를 경로에 추가하여 app 모듈을 임포트할 수 있도록 함
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# app.separator에서 필요한 함수들을 임포트
try:
    from app.separator import separate_audio, mel_spectrogram
except ImportError as e:
    print(f"Error: {e}")
    print("Could not import from 'app.separator'. Make sure you are running this script from the project root directory.")
    sys.exit(1)


# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_and_save(audio_path, output_dir, segment_duration=5.0, device='cuda'):
    """
    하나의 오디오 파일을 처리하여 분리, 분할, Mel Spectrogram 변환 후 저장합니다.
    """
    try:
        # 1. Demucs로 악기 분리 (전체 파일 대상)
        separated_stems, sr = separate_audio(
            audio_path=audio_path,
            model_name='htdemucs',
            device=device,
            target_instruments=None
        )
        
        original_filename = os.path.splitext(os.path.basename(audio_path))[0]

        # 2. 각 악기(stem)별로 처리
        for instrument, tensor in separated_stems.items():
            num_samples_per_segment = int(segment_duration * sr)
            if tensor.dim() > 1:
                tensor = tensor.mean(dim=0)

            segments = list(tensor.split(num_samples_per_segment, dim=-1))

            if segments and segments[-1].shape[-1] < num_samples_per_segment / 2:
                segments.pop()

            if not segments:
                continue

            instrument_dir = os.path.join(output_dir, instrument)
            os.makedirs(instrument_dir, exist_ok=True)

            for i, seg in enumerate(segments):
                mel_spec = mel_spectrogram(seg.unsqueeze(0), sample_rate=sr)
                save_filename = f"{original_filename}_{i:03d}.pt"
                save_path = os.path.join(instrument_dir, save_filename)
                torch.save(mel_spec, save_path)

        return True, None

    except Exception as e:
        logging.error(f"'{audio_path}' 처리 중 오류 발생: {e}", exc_info=True)
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="오디오 파일을 전처리하여 학습용 데이터셋을 생성합니다.")
    parser.add_argument('--input-dir', type=str, default="data/raw", help="처리할 원본 오디오 파일이 있는 디렉토리")
    parser.add_argument('--output-dir', type=str, default="data/processed", help="처리된 Mel Spectrogram 텐서를 저장할 디렉토리")
    parser.add_argument('--segment-duration', type=float, default=5.0, help="오디오를 분할할 시간 (초 단위)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="PyTorch 장치 (cuda 또는 cpu)")
    args = parser.parse_args()

    logging.info(f"입력 디렉토리: {args.input_dir}, 출력 디렉토리: {args.output_dir}")
    
    audio_files = []
    for ext in ["*.mp3", "*.wav", "*.flac"]:
        audio_files.extend(glob(os.path.join(args.input_dir, ext)))

    if not audio_files:
        logging.warning(f"'{args.input_dir}'에서 오디오 파일을 찾을 수 없습니다.")
        return

    logging.info(f"총 {len(audio_files)}개의 오디오 파일을 처리합니다.")
    
    for audio_file in tqdm(audio_files, desc="오디오 파일 처리 중"):
        process_and_save(audio_file, args.output_dir, args.segment_duration, args.device)

    logging.info("--- 전처리 완료 ---")

if __name__ == '__main__':
    main()