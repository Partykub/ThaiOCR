#!/usr/bin/env python3
"""
Thai CRNN Model Architecture
============================

CRNN model implementation for Thai OCR with CTC Loss
Optimized for RTX 5090 GPU training
"""

import torch
import torch.nn as nn
from pathlib import Path


class ThaiCRNN(nn.Module):
    """CRNN Model for Thai OCR with CTC"""
    
    def __init__(self, num_classes, img_height=64, img_width=128, hidden_size=256):
        super(ThaiCRNN, self).__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x64
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x32
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 8x32
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 4x32
        )
        
        # RNN Sequence Modeling
        rnn_input_size = 512 * 4  # 4 is the height after CNN
        self.rnn = nn.LSTM(rnn_input_size, hidden_size, 
                          num_layers=2, bidirectional=True, 
                          batch_first=True, dropout=0.1)
        
        # Classification (including blank token for CTC)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # CNN feature extraction
        batch_size = x.size(0)
        features = self.cnn(x)  # [B, 512, 4, 32]
        
        # Reshape for RNN: [B, sequence_length, feature_size]
        features = features.permute(0, 3, 1, 2)  # [B, 32, 512, 4]
        features = features.contiguous().view(batch_size, 32, -1)
        
        # RNN processing
        rnn_out, _ = self.rnn(features)  # [B, 32, hidden_size*2]
        
        # Classification
        output = self.classifier(rnn_out)  # [B, 32, num_classes]
        
        # For CTC, we need to permute to [T, B, C]
        output = output.permute(1, 0, 2)  # [32, B, num_classes]
        
        return output


def load_char_mapping():
    """Load character mapping from th_dict.txt"""
    dict_file = Path("thai-letters/th_dict.txt")
    
    char_to_idx = {}
    with open(dict_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            char = line.strip()
            if char:  # Skip empty lines
                char_to_idx[char] = idx
    
    # Add blank token for CTC at the end
    char_to_idx['<BLANK>'] = len(char_to_idx)
    
    return char_to_idx


def create_model(char_to_idx):
    """Create ThaiCRNN model with proper number of classes"""
    num_classes = len(char_to_idx)
    model = ThaiCRNN(num_classes=num_classes)
    return model


def load_model(model_path, char_to_idx, device='cuda'):
    """Load trained model from checkpoint"""
    model = create_model(char_to_idx)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model
