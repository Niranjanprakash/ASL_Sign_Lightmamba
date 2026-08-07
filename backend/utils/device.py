import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DEVICE] Selected: {device} | GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"[DEVICE] Selected: {device}")
    return device
