import torch
import torch.nn.functional as F
from tqdm import tqdm

# Device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def Inference(model, loader):
    print(f"Doing inference on {loader}")
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(tqdm(loader, desc="Inference")):
            data = data.to(device) # B, 3, L, W

            logits = model(data) # B, C, L, W
            pred = torch.argmax(logits, dim=1) # B, L, W

            # One-Hot adds a dimension at the end
            pred = F.one_hot(pred, num_classes).permute((0, 3, 1, 2)).float() # B, L, W, C -> B, C, L, W

            