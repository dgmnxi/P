# P프로젝트 - AI 음원 분리 API

> 2025년 2학기 가천대학교 P프로젝트의 AI 코드

## 프로젝트 소개

**핀 포인트 음악 추천 서비스**

[프로젝트 목적 및 개요]

- **개발 기간**: 2025년 2학기
- **소속**: 가천대학교
- **역할**: AI 개발 담당
- **기술 스택**: FastAPI, Demucs, Docker, Google Cloud Platform,FAISS

- 사용자가 선택한 음원의 구간, 악기를 선택하여 해당 구간의 유사한 노래 및 구간을 추천해주는 서비스입니다.

  [ 시연 영상] <br>
  https://www.youtube.com/watch?v=WJ11CrfaiRc

---

##  주요 기능

### AI 음원 분리 API (Demucs)
- YouTube URL을 통한 음원 입력
- AI 기반 음원 트랙 분리 (vocals, drums, bass, other)
- 특정 구간 지정 분리 (start_sec ~ end_sec)

### 변형 ResNet-18 모델을 통한 임베딩 벡터 추출
- 기존 [3,높이,너비]의 입력을 [1,128,431]로 변경
- 마지막 레이어를 FC된 128차원 임베딩 벡터 생성으로 변경
- 변경된 레이어를 제외한 나머지 파라미터들을 전이학습으로 학습

### 생성된 임베딩 벡터를 통한 유사도 검색
- FAISS[https://github.com/facebookresearch/faiss]를 통해 고속 벡터 검색

### API Endpoint

**`POST /recommend`**

입력 파라미터:
```json
{
  "youtube_url": "string",
  "instrument": "string",  // vocals, drums, bass, other
  "start_sec": "float",
  "end_sec": "float",
  "top_k":  "int"  // Optional, Default: 5
}
```

---

## 🔧 작동 방식

### 1. [전체 아키텍처]
[시스템 아키텍처 다이어그램 및 설명 작성]

### 2. [AI 음원 분리 및 전처리 프로세스](app/separator.py , app/prepare_data.py)
- 입력 받은 노래(.mp3)파일을 DEMUCS라이브러리를 통해 4개의 트랙으로 분할
- 각 트랙의 오디오를 5초크기의 batch로 분할
- 각 batch를 Mel-Spectorgram에 통과시켜 [1,128,431]의 텐서 생성 (5초,128 Mel-Filter 기준)
---

## 🚀 배포 방법 (Google Cloud)

### Option A: Compute Engine (GPU 권장)

1. **VM 생성**
   - CPU: `e2-standard-4` 이상
   - GPU: `n1-standard-8` + `T4` / `L4` / `A100`
   - OS: Ubuntu 22.04

2. **Docker 설치**
   ```bash
   # https://docs.docker.com/engine/install/ubuntu/
   ```

3. **GPU VM:  NVIDIA 드라이버 및 Container Toolkit 설치**
   - [Drivers](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)
   - [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

4. **프로젝트 파일 복사**
   ```bash
   gcloud compute scp --recurse .  VM_NAME:~/project
   # 또는 Git clone
   ```

5. **빌드 & 실행**
   ```bash
   # CPU
   docker build -t demucs-api:cpu -f Dockerfile . 
   docker run -d -p 8080:8080 demucs-api:cpu
   
   # GPU
   docker build -t demucs-api: gpu -f Dockerfile.gpu . 
   docker run -d -p 8080:8080 --gpus all demucs-api:gpu
   ```

6. **방화벽 규칙 설정** (TCP:8080 포트 개방)

---

##  참고사항

- 첫 실행 시 Demucs 모델 자동 다운로드
- 대용량 파일 처리 시 GCS 활용 권장
- 프로덕션 환경에서는 Uvicorn worker 수 조정 (`--workers 2`)

---


