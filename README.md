# RPNet - License Plate Detection and Recognition

RPNet is a deep learning-based project for Chinese license plate detection and recognition. This implementation uses a Region Proposal Network (RPNet) architecture with custom ROI pooling for accurate localization and multi-character classification. The project includes standalone inference scripts, pretrained model weights, and comprehensive evaluation tools.

## Features

- **End-to-End License Plate Recognition**: Simultaneous detection and recognition of Chinese license plates
- **Region Proposal Network**: Custom RPNet architecture for accurate bounding box prediction
- **Multi-Scale ROI Pooling**: Advanced feature extraction at multiple scales
- **7-Character Recognition**: Supports full Chinese license plate format (province + letter + 5 alphanumeric)
- **Standalone Inference**: Self-contained script with no external dependencies on project files
- **GPU & CPU Support**: Flexible deployment on both GPU and CPU environments
- **Pretrained Models**: Ready-to-use model weights trained on CCPD dataset

## Model Architecture

The system consists of two main components:

1. **wR2 Backbone**: Deep convolutional network for feature extraction (10 layers)
2. **fh02 Detection Model**: Combined localization and recognition network with:
   - Bounding box regression (4 coordinates)
   - 7 character classifiers for license plate recognition
   - Multi-scale ROI pooling for robust feature extraction

## Directory Structure

```
rpnet/
├── inference.py          # Standalone inference script (recommended)
├── requirements.txt      # Python dependencies
├── models/              # Pretrained model weights
│   ├── fh02.pth        # Main detection & recognition model (210 MB)
│   └── wR2.pth         # Backbone feature extractor (28 MB)
├── orig_src/           # Original source code modules
│   ├── rpnet.py       # Core RPNet implementation
│   ├── demo.py        # Demo script
│   └── rpnetEval.py   # Evaluation utilities
├── demo/              # Sample images for testing
├── outputs/           # Output directory for results
└── docs/              # Documentation
```

## Installation

### Prerequisites
- Python 3.6+
- PyTorch 1.0+
- OpenCV
- NumPy

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rpnet
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pretrained models:**

   The models are available from the CCPD project:

   **fh02.pth** (Main detection & recognition model):
   - [Google Drive](https://drive.google.com/open?id=1YYVWgbHksj25vV6bnCX_AWokFjhgIMhv)
   - [BaiduYun](https://pan.baidu.com/s/1sA-rzn4Mf33uhh1DWNcRhQ)

   **wR2.pth** (Backbone feature extractor):
   - [Google Drive](https://drive.google.com/open?id=1l_tIt7D3vmYNYZLOPbwx8qJpPVM82CP-)
   - [BaiduYun](https://pan.baidu.com/s/1Q3fPDHFYV5uibWwIQxPEOw)

   Place the downloaded models in the `models/` directory.

## Usage

### Basic Inference

The `inference.py` script is a standalone implementation that contains all necessary code:

```bash
# Process a single image
python inference.py -i demo/test.jpg -m models/fh02.pth

# Process a folder of images
python inference.py -i demo/ -m models/fh02.pth

# Save results to a specific output folder
python inference.py -i demo/ -m models/fh02.pth -o outputs/

# Force CPU usage (even if GPU is available)
python inference.py -i demo/ -m models/fh02.pth --cpu

# Run without saving annotated images
python inference.py -i demo/ -m models/fh02.pth --no-save
```

### Command-Line Arguments

- `-i, --input`: Path to input image or folder (required)
- `-m, --model`: Path to model weights file (required)
- `-o, --output`: Output folder for annotated images (default: overwrite input)
- `--no-save`: Do not save annotated images
- `--cpu`: Force CPU usage even if GPU is available

### Output Format

The script outputs:
- **Annotated images**: Original images with bounding boxes and recognized text
- **Console output**: Detected license plate text for each image
- **Processing statistics**: Total time and average time per image

Example output:
```
✓ demo/test.jpg
  Plate: 京A12345 (A12345)
  Saved: outputs/test.jpg
```

## Dataset & Training

This project uses models trained on the **CCPD (Chinese City Parking Dataset)** dataset, which contains over 300,000 diverse Chinese license plate images with comprehensive annotations.

### CCPD Dataset

The CCPD dataset includes:
- **300k+ images** with diverse scenarios
- **Multiple subsets**: Base, DB (dark/bright), Blur, FN (far/near), Rotate, Tilt, Challenge, Green (new energy vehicles)
- **Rich annotations**: Bounding boxes, vertices, tilt angles, brightness, blurriness

**Download CCPD Dataset:**
- [Google Drive](https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc)
- [BaiduYun (code: hm0u)](https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw)

**CCPD-Green (New Energy Vehicles):**
- [Google Drive](https://drive.google.com/file/d/1m8w1kFxnCEiqz_-t2vTcgrgqNIv986PR/view?usp=sharing)
- [BaiduYun (code: ol3j)](https://pan.baidu.com/s/1JSpc9BZXFlPkXxRK4qUCyw)

## Performance Notes

> **Note**: The RPNet architecture is considered old-fashioned by modern standards. For state-of-the-art performance, consider using modern detectors like FCOS or YOLO-based approaches. However, RPNet remains a solid baseline for license plate recognition tasks.

The provided models were trained for only 5 epochs but are sufficient for testing and demonstration purposes.

## Credits & Citation

This project uses the **CCPD (Chinese City Parking Dataset)** and RPNet architecture from:

**Original Repository**: [detectRecog/CCPD](https://github.com/detectRecog/CCPD)

**Paper**: "Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline" (ECCV 2018)

If you use this code or the CCPD dataset in your research, please cite:

```bibtex
@inproceedings{xu2018towards,
  title={Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline},
  author={Xu, Zhenbo and Yang, Wei and Meng, Ajin and Lu, Nanxue and Huang, Huan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2018}
}
```

**Contact**: For questions about the CCPD dataset, contact [detectrecog@gmail.com](mailto:detectrecog@gmail.com)

## License

Please refer to the original [CCPD repository](https://github.com/detectRecog/CCPD) for license information.

## Acknowledgments

- Thanks to the CCPD team for providing the comprehensive dataset and baseline models
- Original RPNet implementation from the CCPD project
- This repository provides a standalone, easy-to-use version of the inference pipeline
