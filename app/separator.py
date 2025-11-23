import librosa
import numpy as np
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from typing import Dict, List, Optional, Tuple

# 백엔드에서 받는 정보들-----------------
#오디오 경로 1. 유튜브에서 직접 가져올 수도, 또는 사용자 음원을 전송할 수 도 있음
audio_path = "Besomorph - Running Cold.mp3"
model_name = "htdemucs"
device = "cuda"
start_time = 30.0  #초 단위
end_time = 35.0    #초 단위
target_instruments = ['vocals', 'drums']  #사용자가 선택한 악기들

#---------------------------------------

#오디오를 로드,시작시간과 종료시간에 맞게 자르기(오디오경로, 시작시간, 종료시간)
def audio_load_and_crop(audio_path, start_time, end_time):
    y, sr = librosa.load(audio_path, sr=44100)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    y_segment = y[start_sample:end_sample]
    return y_segment, sr

# 사용자 요청 오디오 분리 / 오디오 경로,모델이름(optional),장치(optional),타겟 악기(list),시작시간,종료시간(optional)
def separate_audio(
    audio_path: str,
    model_name: str = 'htdemucs',
    device: str = 'cuda',
    target_instruments: Optional[List[str]] = None,
    start_time: float = None,
    end_time: float = None,
) -> Tuple[Dict[str, torch.Tensor], int]:

    model = get_model(name=model_name)
    model.to(device)
    model.eval()

    # Load audio using torchaudio
    wav, sr = torchaudio.load(audio_path)
    wav = wav.to(device)

    # Demucs expects stereo or mono. Let's ensure input is at least 2D (channels, samples)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    # Crop audio
    if end_time > start_time:
        start_frame = int(start_time * sr)
        end_frame = int(end_time * sr)
        wav = wav[..., start_frame:end_frame]

    # Apply model - demucs expects a batch dimension, so add it
    with torch.no_grad():
        estimates = apply_model(model, wav.unsqueeze(0))[0]  # Take first item from batch

    # Filter for target instruments
    results = {}
    for i, source_name in enumerate(model.sources):
        if target_instruments is None or source_name in target_instruments:
            results[source_name] = estimates[i].cpu()

    return results, sr

#mel 스펙토그램 계산 / 오디오텐서만 넘겨도 됨
def mel_spectrogram(
    audio_tensor: torch.Tensor,
    sample_rate: int = 44100,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Computes the Mel spectrogram of an audio tensor.
    """
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_spec = mel_spectrogram_transform(audio_tensor)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    return mel_spec_db

# 실제 분리 

results, sr = separate_audio(
    audio_path=audio_path,
    model_name=model_name,
    device=device,
    target_instruments=target_instruments,
    start_time=start_time,
    end_time=end_time)



# If multi-channel, convert to mono by averaging channels, and move to CPU
if vocals.dim() == 2:
    # vocals shape: (channels, samples) -> convert to (1, samples)
    audio_tensor = vocals.mean(dim=0, keepdim=True).cpu()
else:
    audio_tensor = vocals.cpu()


mel_result = mel_spectrogram(audio_tensor, sample_rate=sr)

# Mel 스펙토그램 텐서의 차원(shape)을 출력 ( 1x128x431)
print(f"Mel Spectrogram Shape: {mel_result.shape}")