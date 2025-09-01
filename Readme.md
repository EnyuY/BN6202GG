# Gesture Control Gaming System

## Gesture Mapping

- **Accelerate**: Both hands closed (fists) → W key
- **Brake**: Both hands open → Spacebar
- **Reverse**: Right hand open + left hand closed → S key
- **Start**: Right hand closed + left hand open → E key
- **Steering**: Hand height difference controls A/D keys

## Quick Start

### MediaPipe Version (Recommended)
```bash
pip install -r requirements_mediapipe.txt
python GGrun_MEDIAPIPE.py
```

### Full Version
```bash
pip install -r requirements.txt  
python GGrun_PYTORCH.py
```

## Version Overview

- `GGrun_MEDIAPIPE.py` - Lightweight version, no PyTorch required
- `GGrun_PYTORCH.py` - Full version with CUDA acceleration support
- `GGrun_CPU.py` - CPU-only version

## System Requirements

- Python 3.10+
- Windows 10/11
- OBS Studio

## Configuration

Modify `config.json` to adjust camera, gesture recognition, and other parameters.