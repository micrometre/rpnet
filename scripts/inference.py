#!/usr/bin/env python3
"""
Standalone License Plate Detection and Recognition Script
This script contains all necessary code embedded and doesn't rely on external Python files.
Only requires: PyTorch, OpenCV, NumPy, and the model weights file (fh02.pth)

Usage:
    python inference.py -i demo/ -m fh02.pth --cpu -o output
    
"""

import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import numpy as np
import os
import re
from glob import glob
from time import time


# ============================================================================
# Configuration - Consolidates all hyperparameters
# ============================================================================

CONFIG = {
    'wR2_fc_size': 23232,
    'fh02_fc_size': 53248,
    'fh02_hidden': 128,
    'fh02_roi_size': (16, 8),
    'img_size': (480, 480),
    'dropout': 0.2,
}


# ============================================================================
# ROI Pooling Implementation (embedded from roi_pooling.py)
# ============================================================================

def adaptive_max_pool(input, size):
    """Adaptive max pooling for ROI"""
    return F.adaptive_max_pool2d(input, (size[1], size[0]))


def roi_pooling_ims(input, rois, size=(7, 7), spatial_scale=1.0):
    """
    ROI pooling for images - one ROI per image
    Args:
        input: feature maps
        rois: regions of interest (N x 4) [x1, y1, x2, y2]
        size: output size (w, h)
        spatial_scale: scale factor
    """
    assert rois.dim() == 2
    assert len(input) == len(rois)
    assert rois.size(1) == 4
    
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)
    
    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    
    for i in range(num_rois):
        roi = rois[i]
        im = input.narrow(0, i, 1)[..., roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1)]
        output.append(adaptive_max_pool(im, size))
    
    return torch.cat(output, 0)


# ============================================================================
# Model Architecture: wR2 (embedded from wR2.py and demo.py)
# ============================================================================

class wR2(nn.Module):
    """Backbone network for feature extraction"""
    def __init__(self, num_classes=1000):
        super(wR2, self).__init__()
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8,
            hidden9,
            hidden10
        )
        self.classifier = nn.Sequential(
            nn.Linear(23232, 100),
            nn.Linear(100, 100),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.classifier(x)


# ============================================================================
# Model Architecture: fh02 (main detection + recognition model)
# ============================================================================

class fh02(nn.Module):
    """
    License plate detection and recognition model
    Outputs: bounding box location + 7 character classifications
    """
    def __init__(self, num_points, num_classes, provNum=38, alphaNum=25, adNum=35):
        super(fh02, self).__init__()
        self.wR2 = wR2(num_points)
        self.wR2 = torch.nn.DataParallel(self.wR2, device_ids=range(torch.cuda.device_count()))
        
        # 7 classifiers: [province, alphabet, alphanumeric x5]
        self.classifier1 = nn.Sequential(
            nn.Linear(53248, CONFIG['fh02_hidden']),
            nn.Linear(CONFIG['fh02_hidden'], provNum),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(53248, CONFIG['fh02_hidden']),
            nn.Linear(CONFIG['fh02_hidden'], alphaNum),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(53248, CONFIG['fh02_hidden']),
            nn.Linear(CONFIG['fh02_hidden'], adNum),
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(53248, CONFIG['fh02_hidden']),
            nn.Linear(CONFIG['fh02_hidden'], adNum),
        )
        self.classifier5 = nn.Sequential(
            nn.Linear(53248, CONFIG['fh02_hidden']),
            nn.Linear(CONFIG['fh02_hidden'], adNum),
        )
        self.classifier6 = nn.Sequential(
            nn.Linear(53248, CONFIG['fh02_hidden']),
            nn.Linear(CONFIG['fh02_hidden'], adNum),
        )
        self.classifier7 = nn.Sequential(
            nn.Linear(53248, CONFIG['fh02_hidden']),
            nn.Linear(CONFIG['fh02_hidden'], adNum),
        )

    def forward(self, x):
        # Feature extraction with skip connections
        x0 = self.wR2.module.features[0](x)
        _x1 = self.wR2.module.features[1](x0)
        x2 = self.wR2.module.features[2](_x1)
        _x3 = self.wR2.module.features[3](x2)
        x4 = self.wR2.module.features[4](_x3)
        _x5 = self.wR2.module.features[5](x4)

        x6 = self.wR2.module.features[6](_x5)
        x7 = self.wR2.module.features[7](x6)
        x8 = self.wR2.module.features[8](x7)
        x9 = self.wR2.module.features[9](x8)
        x9 = x9.view(x9.size(0), -1)
        boxLoc = self.wR2.module.classifier(x9)

        # ROI pooling with feature maps from intermediate layers
        h1, w1 = _x1.data.size()[2], _x1.data.size()[3]
        device = _x1.device
        p1 = Variable(torch.FloatTensor([[w1,0,0,0],[0,h1,0,0],[0,0,w1,0],[0,0,0,h1]]).to(device), requires_grad=False)
        
        h2, w2 = _x3.data.size()[2], _x3.data.size()[3]
        p2 = Variable(torch.FloatTensor([[w2,0,0,0],[0,h2,0,0],[0,0,w2,0],[0,0,0,h2]]).to(device), requires_grad=False)
        
        h3, w3 = _x5.data.size()[2], _x5.data.size()[3]
        p3 = Variable(torch.FloatTensor([[w3,0,0,0],[0,h3,0,0],[0,0,w3,0],[0,0,0,h3]]).to(device), requires_grad=False)

        # Convert box format: x, y, w, h --> x1, y1, x2, y2
        assert boxLoc.data.size()[1] == 4
        postfix = Variable(torch.FloatTensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]).to(device), requires_grad=False)
        boxNew = boxLoc.mm(postfix).clamp(min=0, max=1)

        # ROI pooling on three scale feature maps
        roi1 = roi_pooling_ims(_x1, boxNew.mm(p1), size=(16, 8))
        roi2 = roi_pooling_ims(_x3, boxNew.mm(p2), size=(16, 8))
        roi3 = roi_pooling_ims(_x5, boxNew.mm(p3), size=(16, 8))
        rois = torch.cat((roi1, roi2, roi3), 1)

        _rois = rois.view(rois.size(0), -1)

        # Character classification
        y0 = self.classifier1(_rois)
        y1 = self.classifier2(_rois)
        y2 = self.classifier3(_rois)
        y3 = self.classifier4(_rois)
        y4 = self.classifier5(_rois)
        y5 = self.classifier6(_rois)
        y6 = self.classifier7(_rois)
        
        return boxLoc, [y0, y1, y2, y3, y4, y5, y6]


# ============================================================================
# Character Mappings
# ============================================================================

PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", 
             "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", 
             "新", "警", "学", "O"]

ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 
             'U', 'V', 'W', 'X', 'Y', 'Z', 'O']

ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 
       'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


# ============================================================================
# Image Processing Functions
# ============================================================================

def preprocess_image(image_path, img_size=(480, 480)):
    """
    Load and preprocess an image for inference
    Args:
        image_path: path to image file
        img_size: target size (width, height)
    Returns:
        preprocessed tensor, original image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize and normalize
    resized = cv2.resize(img, img_size)
    # Convert to CHW format
    tensor = np.transpose(resized, (2, 0, 1))
    tensor = tensor.astype('float32') / 255.0
    
    # Add batch dimension
    tensor = np.expand_dims(tensor, axis=0)
    
    return torch.from_numpy(tensor), img


def draw_result(image, box_coords, plate_text, color=(0, 0, 255), thickness=2):
    """
    Draw bounding box and text on image
    Args:
        image: original image (numpy array)
        box_coords: [cx, cy, w, h] normalized coordinates
        plate_text: recognized plate text
        color: BGR color tuple
        thickness: line thickness
    Returns:
        annotated image
    """
    img = image.copy()
    cx, cy, w, h = box_coords
    
    # Convert normalized coords to pixel coords
    img_h, img_w = img.shape[:2]
    left_up = [int((cx - w/2) * img_w), int((cy - h/2) * img_h)]
    right_down = [int((cx + w/2) * img_w), int((cy + h/2) * img_h)]
    
    # Draw rectangle
    cv2.rectangle(img, tuple(left_up), tuple(right_down), color, thickness)
    
    # Draw text
    cv2.putText(img, plate_text, (left_up[0], left_up[1] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness)
    
    return img


def get_image_files(input_path):
    """
    Get list of image files from path (file or directory)
    """
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            for pattern in [f'*.{ext}', f'*.{ext.upper()}']:
                image_files.extend(glob(os.path.join(input_path, pattern)))
        return sorted(image_files)
    else:
        raise ValueError(f"Invalid path: {input_path}")


# ============================================================================
# Inference Class
# ============================================================================

class LicensePlateDetector:
    """Standalone license plate detector and recognizer"""
    
    def __init__(self, model_path, use_gpu=None):
        """
        Initialize detector
        Args:
            model_path: path to model weights (.pth file)
            use_gpu: whether to use GPU (auto-detect if None)
        """
        self.img_size = (480, 480)
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
        
        print(f"Using device: {'GPU' if self.use_gpu else 'CPU'}")
        
        # Load model
        self.model = fh02(num_points=4, num_classes=4)
        self.model = torch.nn.DataParallel(
            self.model, 
            device_ids=range(torch.cuda.device_count()) if self.use_gpu else []
        )
        
        # Load weights with state_dict conversion
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = self._convert_state_dict(state_dict)
        self.model.load_state_dict(state_dict)
        
        if self.use_gpu:
            self.model = self.model.cuda()
        
        self.model.eval()
        print(f"Model loaded from: {model_path}")
    
    def _convert_state_dict(self, state_dict):
        """
        Convert state_dict if needed (in case of format mismatches)
        The checkpoint should already have module.classifier1, module.classifier2, etc.
        """
        # Just return as-is, the checkpoint format matches the model
        return state_dict
    
    def detect(self, image_path):
        """
        Detect and recognize license plate in image
        Args:
            image_path: path to image file
        Returns:
            dict with keys: 'plate_text', 'box_coords', 'confidence'
        """
        # Preprocess
        img_tensor, original_img = preprocess_image(image_path, self.img_size)
        
        # Move to GPU if available
        if self.use_gpu:
            x = Variable(img_tensor.cuda(0))
        else:
            x = Variable(img_tensor)
        
        # Inference
        with torch.no_grad():
            fps_pred, y_pred = self.model(x)
        
        # Decode predictions
        predictions = [el.data.cpu().numpy()[0] for el in y_pred]
        pred_indices = [np.argmax(p) for p in predictions]
        box_coords = fps_pred.data.cpu().numpy()[0].tolist()
        
        char_sets = [PROVINCES, ALPHABETS] + [ADS] * 5
        plate_chars = [char_set[idx] for char_set, idx in zip(char_sets, pred_indices)]
        plate_text_full = ''.join(plate_chars)
        plate_text = ''.join(plate_chars[1:])
        
        return {
            'plate_text': plate_text,
            'plate_text_full': plate_text_full,
            'box_coords': box_coords,
            'original_image': original_img
        }
    
    def process_image(self, image_path, output_path=None, save_annotated=True):
        """
        Process single image and optionally save result
        Args:
            image_path: path to input image
            output_path: path to save annotated image (optional)
            save_annotated: whether to save annotated image
        Returns:
            result dict
        """
        result = self.detect(image_path)
        
        if save_annotated:
            # Draw results on image
            annotated_img = draw_result(
                result['original_image'],
                result['box_coords'],
                result['plate_text']
            )
            
            # Determine output path
            if output_path is None:
                output_path = image_path  # Overwrite original
            
            cv2.imwrite(output_path, annotated_img)
            result['output_path'] = output_path
        
        return result


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Standalone License Plate Detection and Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image (overwrites original)
  python inference_standalone.py -i test.jpg -m fh02.pth
  
  # Process folder of images
  python inference_standalone.py -i demo/ -m fh02.pth
  
  # Save to output folder
  python inference_standalone.py -i demo/ -m fh02.pth -o results/
  
  # Process without saving annotated images
  python inference_standalone.py -i test.jpg -m fh02.pth --no-save
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Path to input image or folder')
    parser.add_argument('-m', '--model', required=True,
                        help='Path to model weights file (.pth)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output folder for annotated images (default: overwrite input)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save annotated images')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file does not exist: {args.model}")
        return
    
    # Create output directory if specified
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Created output directory: {args.output}")
    
    # Initialize detector
    use_gpu = not args.cpu
    detector = LicensePlateDetector(args.model, use_gpu=use_gpu)
    
    # Get image files
    image_files = get_image_files(args.input)
    print(f"\nFound {len(image_files)} image(s) to process\n")
    
    # Process images
    start_time = time()
    results = []
    
    for img_path in image_files:
        try:
            # Determine output path
            if args.output:
                filename = os.path.basename(img_path)
                output_path = os.path.join(args.output, filename)
            else:
                output_path = img_path
            
            # Process
            result = detector.process_image(
                img_path, 
                output_path=output_path,
                save_annotated=not args.no_save
            )
            
            results.append(result)
            
            # Print result
            print(f"✓ {img_path}")
            print(f"  Plate: {result['plate_text_full']} ({result['plate_text']})")
            if not args.no_save:
                print(f"  Saved: {result['output_path']}")
            print()
            
        except Exception as e:
            print(f"✗ {img_path}")
            print(f"  Error: {str(e)}\n")
    
    # Summary
    elapsed = time() - start_time
    print(f"{'='*60}")
    print(f"Processed {len(results)}/{len(image_files)} images successfully")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average: {elapsed/len(image_files):.2f} seconds per image")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
