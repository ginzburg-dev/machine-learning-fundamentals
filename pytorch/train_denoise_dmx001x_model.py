# train_denoiser.py
import sys
import argparse
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=10)
    return p.parse_args()

class SimpleDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

def print_system_information():
    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        print("GPU 0:", torch.cuda.get_device_name(0))

def load_image(path, device):
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()
    x = t(img).unsqueeze(0).to(device)  # [1, C, H, W]
    return x

def save_image(tensor, path):
    t = transforms.ToPILImage()
    # Clamp to valid display range
    img_tensor = tensor.detach().cpu().squeeze(0)
    img_tensor = img_tensor.clamp(0.0, 1.0)
    img = t(img_tensor)
    img.save(path)

def main():
    args = parse_args()
    
    # 1) Choose device -> this is what makes it “GPU on farm”
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2) Load data
    x_noisy = load_image(args.input, device)
    x_clean = load_image(args.target, device)

    # 3) Model + optimizer
    model = SimpleDenoiser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 4) Training loop (toy)
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        out = model(x_noisy)
        loss = loss_fn(out, x_clean)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{args.epochs}, loss = {loss.item():.6f},",
              f"PROGRESS: {int(((epoch+1)/args.epochs)*100)}%")
        sys.stdout.flush()

    # 5) Save denoised result
    with torch.no_grad():
        out = model(x_noisy)
    save_image(out, args.output)

if __name__ == "__main__":
    main()