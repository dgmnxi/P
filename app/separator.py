'''
 오디오 분리 및 Mel Spectrogram 변환 기능을 제공합니다.
 
 '''

import librosa
import numpy as np
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from typing import Dict, List, Optional, Tuple

# --- Mel Spectrogram 변환 함수---
def mel_spectrogram(
    audio_tensor: torch.Tensor,
    sample_rate: int = 44100,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """오디오 텐서를 Mel Spectrogram으로 변환합니다."""

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec = mel_transform(audio_tensor)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    return mel_spec_db

# --- 오디오 분리 함수 (최종 수정) ---
def separate_audio(
    audio_path: str,
    model_name: str = 'htdemucs',
    device: str = 'cuda',
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    오디오 파일을 분리합니다.
    1. 오디오를 로드하고 'demucs'가 요구하는 스테레오(2채널)로 변환합니다.
    2. 'demucs'로 악기를 분리합니다.
    3. 분리된 스테레오 트랙을 그대로 반환합니다.
    """
    model = get_model(name=model_name)
    model.to(device)
    model.eval()

    wav, sr = torchaudio.load(audio_path)
    wav = wav.to(device)

    # 1. demucs 입력을 위해 스테레오(2채널) 보장
    # 원본이 모노(1채널)이면 채널을 복제해서 스테레오로 만듭니다.
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    # 원본이 2채널 초과이면(예: 5.1채널) 강제로 스테레오로 만듭니다.
    elif wav.shape[0] > 2:
        wav = wav[:2, :]

    # 2. demucs로 분리 (입력은 [배치, 채널, 샘플] 형태여야 함)
    with torch.no_grad():
        # wav는 이제 [2, 샘플 수] 이므로, 배치 차원을 추가하여 [1, 2, 샘플 수]로 만듭니다.
        estimates = apply_model(model, wav.unsqueeze(0), overlap=0.5)[0]

    # 3. 분리된 스테레오 트랙을 그대로 반환
    results = {name: estimates[i].cpu() for i, name in enumerate(model.sources)}
    return results, sr



# --- API 서버용 함수 (main.py에서 사용) ---
# TODO:하나의 악기만 받을 수 있음 아직
def extract_and_transform_frame(
    audio_path: str,
    instrument: str,
    start_sec: float,
    end_sec: float,
    device: str = 'cuda'
) -> Optional[torch.Tensor]:
    """
    오디오 파일의 특정 구간, 특정 악기를 추출하여 Mel Spectrogram으로 변환합니다.
    """
    try:
        model = get_model(name='htdemucs')
        model.to(device)
        model.eval()

        wav, sr = torchaudio.load(audio_path)
        wav = wav.to(device)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # 오디오 자르기
        start_frame = int(start_sec * sr)
        end_frame = int(end_sec * sr)
        wav_segment = wav[..., start_frame:end_frame]

        # Demucs로 분리
        with torch.no_grad():
            estimates = apply_model(model, wav_segment.unsqueeze(0), overlap=0.5)[0]

        # 원하는 악기 찾기
        source_idx = model.sources.index(instrument)
        instrument_wav = estimates[source_idx].cpu()

        # 모노 변환 및 Mel Spectrogram 계산
        if instrument_wav.dim() > 1:
            instrument_wav = instrument_wav.mean(dim=0, keepdim=True)
        
        return mel_spectrogram(instrument_wav, sample_rate=sr)

    except Exception as e:
        print(f"Error in extract_and_transform_frame: {e}")
        return None