import os
import shutil
import tempfile
import zipfile
from pathlib import Path


import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse

# separator 모듈을 직접 import
from .separator import separate_audio

app = FastAPI(title="Demucs Server", version="1.0.0")


def ensure_ffmpeg_on_path(project_root: Path) -> None:
    # If a portable ffmpeg exists under tools/ffmpeg/**/bin, prepend to PATH
    tools_dir = project_root / "tools" / "ffmpeg"
    if tools_dir.exists():
        for p in tools_dir.rglob("bin/ffmpeg"):
            bin_dir = p.parent
            if str(bin_dir) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}" + os.environ.get("PATH", "")
                break
        for p in tools_dir.rglob("bin/ffmpeg.exe"):
            bin_dir = p.parent
            if str(bin_dir) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}" + os.environ.get("PATH", "")
                break


def resolve_device(requested: str) -> str:
    req = (requested or "auto").lower()
    if req == "cpu":
        return "cpu"
    if req == "cuda":
        try:
            import torch  # type: ignore
            if torch.cuda.is_available() and getattr(torch.version, "cuda", None):
                return "cuda"
        except Exception:
            pass
        return "cpu"
    # auto
    try:
        import torch  # type: ignore
        if torch.cuda.is_available() and getattr(torch.version, "cuda", None):
            return "cuda"
    except Exception:
        pass
    return "cpu"


@app.get("/healthz")
def healthz():
    try:
        import demucs  # type: ignore
        return {"status": "ok", "demucs": getattr(demucs, "__version__", "unknown")}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

@app.post("/separate")
async def separate(
    file: UploadFile = File(...),
    start_time: Optional[float] = Form(None),
    end_time: Optional[float] = Form(None),
    device: str = Form("auto"),
    model: str = Form("htdemucs"),
    instruments: Optional[List[str]] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    workdir = Path(tempfile.mkdtemp(prefix="demucs_"))
    tmp_path = workdir / file.filename
    
    try:
        # Save upload to a temporary file
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Run separation
        separated_tensors, sr = separate_audio(
            audio_path=str(tmp_path),
            model_name=model,
            device=resolve_device(device),
            target_instruments=instruments,
            start_time=start_time,
            end_time=end_time,
        )

        # Save separated audio files
        stems_dir = workdir / "stems"
        stems_dir.mkdir()
        for name, tensor in separated_tensors.items():
            # Ensure tensor is 2D (channels, samples) for torchaudio.save
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            torchaudio.save(stems_dir / f"{name}.wav", tensor, sr) # Create a zip file of the results zip_path = workdir / "separated_stems.zip" with zipfile.ZipFile(zip_path, 'w') as zipf: for stem_file in stems_dir.glob("*.wav"):
                zipf.write(stem_file, arcname=stem_file.name)
        
        return FileResponse(zip_path, media_type="application/zip", filename=f"{Path(file.filename).stem}_separated.zip")

    except Exception as e:
        # Log the full error for debugging
        print(f"Error during separation: {e}")
        raise HTTPException(status_code=500, detail=f"Separation failed: {str(e)}")
    finally:
        # Cleanup the temporary directory
        shutil.rmtree(workdir, ignore_errors=True)
        try:
            await file.close()
        except Exception:
            pass

