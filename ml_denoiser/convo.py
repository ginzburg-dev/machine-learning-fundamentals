import sys
import torch # type: ignore[reportMissingImports]
import torch.nn as nn # type: ignore[reportMissingImports]
import torch.optim as optim
from torchvision import transforms as T # type: ignore[reportMissingImports]
from PIL import Image # type: ignore[reportMissingImports]

import logging

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("GPU 0:", torch.cuda.get_device_name(0))

# ---------------------------
# 1. Load images as tensors
# ---------------------------

to_tensor = T.ToTensor()        # [0,255] -> [0,1], HWC -> CHW
to_pil = T.ToPILImage()

noisy_img = Image.open("pytorch/images/noisy.png").convert("RGB")
clean_img = Image.open("pytorch/images/clean.png").convert("RGB")

noisy = to_tensor(noisy_img).unsqueeze(0)   # shape: [1, 3, H, W]
clean = to_tensor(clean_img).unsqueeze(0)   # shape: [1, 3, H, W]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
device = torch.device(device)
noisy = noisy.to(device)
clean = clean.to(device)

# ---------------------------
# 2. Very simple CNN model
# ---------------------------

class SimpleDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

model = SimpleDenoiser().to(device)

# ---------------------------
# 3. Loss + optimizer
# ---------------------------

criterion = nn.MSELoss()              # pixel-wise L2
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 4. Training loop (toy)
# ---------------------------

num_epochs = 300

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    output = model(noisy)             # predict clean from noisy
    loss = criterion(output, clean)   # compare to ground truth

    loss.backward()                   # dLoss/dWeights
    optimizer.step()                  # update weights

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, loss = {loss.item():.6f},",
            f"PROGRESS: {int(((epoch+1)/num_epochs)*100)}%")
        sys.stdout.flush()

# --- Inspect first conv layer weights ---

first_conv = model.net[0]  # nn.Conv2d(3, 32, 3, padding=1)
w = first_conv.weight.data  # shape: [32, 3, 3, 3]

print("First conv weight shape:", w.shape)

# print first filter for the R channel (out_ch=0, in_ch=0)
print("Kernel[0, 0, :, :]:")
print(w[0, 0, :, :]) # type: ignore

# ---------------------------
# 5. Run model and save result
# ---------------------------

model.eval()
with torch.no_grad():
    denoised = model(noisy)
    denoised = torch.clamp(denoised, 0.0, 1.0)   # keep valid [0,1] range

denoised_img = to_pil(denoised.squeeze(0).cpu())
denoised_img.save("pytorch/images/denoised.png")

print("Saved denoised image as denoised.png")
