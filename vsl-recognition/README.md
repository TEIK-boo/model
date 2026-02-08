# VSL Recognition - Sign Language Recognition

CNN + LSTM model for Vietnamese Sign Language recognition using MediaPipe keypoints.

## Quick Start

```bash
# 1. Prepare dataset (extract + augment)
python -m src.data.prepare_pipeline

# 2. Train model
python -m src.training.pipeline

# 3. Run inference
python -m src.inference.realtime --mode webcam
```

## Project Structure

```
vsl-recognition/
├── src/                          # Main source code (renamed from preprocessing/)
│   ├── data/                     # Data preparation
│   │   ├── extract.py            # Keypoint extraction
│   │   ├── augment.py            # Data augmentation
│   │   ├── check_distribution.py # Dataset statistics
│   │   └── prepare_pipeline.py   # Full pipeline
│   │
│   ├── models/                   # Model architectures
│   │   ├── components.py         # CNN/MLP branches
│   │   ├── hybrid.py             # MLP+LSTM model
│   │   ├── stateful.py           # Stateful variant
│   │   └── converter.py          # Model converter
│   │
│   ├── training/                 # Training pipeline
│   │   ├── data_loader.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── pipeline.py
│   │
│   ├── inference/                # Inference
│   │   └── realtime.py
│   │
│   ├── visualization/            # Visualization
│   │   ├── keypoints.py
│   │   └── sequences.py
│   │
│   ├── utils/                    # Utilities
│   │   ├── extraction.py
│   │   ├── augmentation.py
│   │   ├── inference_utils.py
│   │   └── viz_utils.py
│   │
│   └── config.py
│
├── data/                         # Dataset
└── requirements.txt
```

## Model Architecture

```
Input (33 frames, 1662 keypoints)
    ↓
MLP Branches → LSTM → Softmax (76 classes)
```

## Usage

### Data Preparation
```bash
python -m src.data.prepare_pipeline
```

### Training
```bash
python -m src.training.pipeline
```

### Inference
```bash
python -m src.inference.realtime --mode webcam
```

## Requirements

```bash
pip install -r requirements.txt
```

---

## Server Deployment

### Configuration

The `config.py` is configured for server paths:
- Data: `/mnt/ngan/vsl_data/VSL_data/`
- Code: `/home/islabworker2/mya/vsl-recognition`
- Sequences: `/mnt/ngan/vsl_data/VSL_data/sequences/`

### Setup on Server

```bash
# 1. Clone/pull code
cd /home/islabworker2/mya/vsl-recognition
git pull

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Training on Server

```bash
# Run training (foreground)
python main.py train

# Or with nohup (background)
nohup python main.py train > training.log 2>&1 &
tail -f training.log
```

### Monitor Training

```bash
# Check processes
ps aux | grep "main.py"

# Monitor GPU
nvidia-smi

# View logs
tail -f training.log
tail -f logs/training_*.log
```

