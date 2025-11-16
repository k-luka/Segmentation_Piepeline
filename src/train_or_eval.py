import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.amp import autocast
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import wandb

# Device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, eps=1e-7, num_classes=3):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        # logtis -> B, C, L, W.  targets -> B, L, W
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0,3,1,2).float() # -> B,C,L,W
        # dice compute
        inter = (probs * targets_oh).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets_oh.sum(dim=(2,3))
        dice = (2 * inter + self.eps) / (union + self.eps)
        dice_loss = 1 - dice.mean()
        loss = (self.alpha * ce_loss) + ((1 - self.alpha) * dice_loss)
        return loss


class Trainer:
    def __init__(self, model, device, criterion, optimizer, scheduler, checkpoint_path, result_dir):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.result_dir = result_dir


    def train_epoch(self, epoch, loader, num_epochs):
        self.model.train()
        for data, target in tqdm(loader, desc=f"Training {epoch+1}/{num_epochs}"):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.device=="cuda")):
                logits = self.model(data)
                loss = self.criterion(logits, target)

            loss.backward()
            self.optimizer.step()
            wandb.log({"train_loss":loss.item()})
        self.scheduler.step()


    def eval(self, loader, loader_name, num_classes=3):
        self.model.eval()

        intersection = torch.zeros(num_classes, device=self.device)
        pred_sum = torch.zeros(num_classes, device=self.device)
        target_sum = torch.zeros(num_classes, device=self.device)

        with torch.no_grad():
            num_correct = 0
            num_total = 0
            for _, (data, target) in enumerate(tqdm(loader, desc=f"Evaluating {loader_name} dataset")):
                data = data.to(self.device) # B, 3, L, W
                target = target.to(self.device).long() # B, L, W
        
                logits = self.model(data) # B, C, L, W
                pred = torch.argmax(logits, dim=1) # B, L, W

                # One-Hot adds a dimension at the end
                pred = F.one_hot(pred, num_classes).permute((0, 3, 1, 2)).float() # B, L, W, C -> B, C, L, W
                target = F.one_hot(target, num_classes).permute((0, 3, 1, 2)).float()

                # add
                intersection += (pred * target).sum(dim=(0,2,3))
                pred_sum += pred.sum(dim=(0,2,3))
                target_sum += target.sum(dim=(0,2,3))

            eps = 1e-7     
            per_class_dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
            mean_dice = per_class_dice.mean().item()

            formatted_per_class = [f"{x:.4f}" for x in per_class_dice]
            print(f"Mean {loader_name} dice: {mean_dice:.4f}. Mean {loader_name} perclass dice: {formatted_per_class}")
        return mean_dice

    def inference_save(self, loader, result_dir, epoch):
        result_dir = Path(self.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        with torch.no_grad():
            preds, imgs = [], []
            for (img, raw_img) in loader:
                img = img.to(self.device) # (B, C, H, W)
                img = img.unsqueeze(0) # add batch dim 
                logits = self.model(img)
                pred_mask = torch.argmax(logits, dim=1).cpu()
                preds.append(pred_mask)
                imgs.append(raw_img)

            
            for idx, (raw, mask) in enumerate(zip(imgs, preds)):
                new_dir = result_dir / f"epoch_{epoch}"
                new_dir.mkdir(parents=True, exist_ok=True)
                raw.save(new_dir / f"image_{idx}.png")
                mask_img = colorize(mask.numpy())
                mask_img.save(new_dir / f"mask_{idx}.png")

    def train(self, train_loader, eval_loader, inference_loader, eval_strat, num_epochs):
        best_mean_dice = 0
        for epoch in range(num_epochs):
            self.train_epoch(epoch, train_loader, num_epochs) # train

            if (epoch+1) % eval_strat == 0:
                mean_dice = self.eval(eval_loader, "eval") # eval
                if mean_dice > best_mean_dice:
                    print(f"Saving new best model at {epoch+1}.")
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    self.inference_save(inference_loader, self.result_dir, epoch) # inference


PALETTE = {
    0: (128, 0, 128),   # purple
    1: (255, 215, 0),   # yellow
    2: (30, 144, 255),  # blue (pick any RGB you like)
}

def colorize(mask):
    mask = mask.squeeze()
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in PALETTE.items():
        rgb[mask == cls] = color
    return Image.fromarray(rgb)


# # train
# def Train(model, num_epochs, loader, criterion, optimizer, scheduler):
#         model.train()
#         for batch_idx, (data, target) in enumerate(tqdm(loader, desc = f"Training {epoch+1}/{num_epochs}")):
#             data = data.to(device)
#             target = target.to(device).long()
#             optimizer.zero_grad()

#             logit = model(data)

#             loss = criterion(logit, target)
#             loss.backward()
#             optimizer.step()
#             wandb.log({"train_loss":loss.item()})
        

# # eval
# def Evaluate(model, loader, loader_name, num_classes):
#     model.eval()

#     intersection = torch.zeros(num_classes, device=device)
#     pred_sum = torch.zeros(num_classes, device=device)
#     target_sum = torch.zeros(num_classes, device=device)

#     with torch.no_grad():
#         num_correct = 0
#         num_total = 0
#         for _, (data, target) in enumerate(tqdm(loader, desc=f"Evaluating {loader_name} dataset")):
#             data = data.to(device) # B, 3, L, W
#             target = target.to(device).long() # B, L, W
#             print("target shape:", target.shape)
    
#             logits = model(data) # B, C, L, W
#             pred = torch.argmax(logits, dim=1) # B, L, W

#             # One-Hot adds a dimension at the end
#             pred = F.one_hot(pred, num_classes).permute((0, 3, 1, 2)).float() # B, L, W, C -> B, C, L, W
            
#             target = F.one_hot(target, num_classes).permute((0, 3, 1, 2)).float()

#             # add
#             intersection += (pred * target).sum(dim=(0,2,3))
#             pred_sum += pred.sum(dim=(0,2,3))
#             target_sum += target.sum(dim=(0,2,3))


#         eps = 1e-7     
#         per_class_dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
#         mean_dice = per_class_dice.mean().item()

#     return mean_dice, per_class_dice.tolist()
