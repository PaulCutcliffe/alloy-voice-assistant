import torch
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Check your PyTorch installation.")