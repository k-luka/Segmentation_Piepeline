from model.UNet import Unet
from src.train_or_eval import Trainer, CombinedLoss
from src.dataset import data_loaders
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_class



# Device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # wandb
    wandb.init(project=cfg.wandb.project)

    # model init
    modelCls = get_class(cfg.model.class_path) # model.UNet.Unet
    model = modelCls(in_channels=cfg.model.in_channels, num_classes=cfg.model.num_classes)
    model.to(device)

    parametersWithDecay = []
    parametersWithoutDecay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name.lower() or "positionalEmbedding" in name:
            parametersWithoutDecay.append(param)
        else:
            parametersWithDecay.append(param)
    parameterGroups = [
        {"params": parametersWithDecay, "weight_decay": cfg.model.optimizer.weight_decay},
        {"params": parametersWithoutDecay, "weight_decay": 0.0}
    ]

    # optimizer, criterion
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(parameterGroups, cfg.train.lr)
    # scheduler
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cfg.train.num_epochs)
    scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=(cfg.train.num_epochs // 10))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[cfg.train.num_epochs // 10])

    # data loader
    train_loader, eval_loader, inference_loader = data_loaders(cfg)

    trainer = Trainer(model, device, criterion, optimizer, scheduler, cfg.model.checkpoint, cfg.model.result_dir)

    trainer.train(train_loader, eval_loader, inference_loader, eval_strat=cfg.train.eval_strat, num_epochs=cfg.train.num_epochs)

    # # train
    # Train(model, cfg.train.num_epochs, train_loader, criterion, optimizer, scheduler)

    # # save
    # torch.save(model.state_dict(), cfg.model.checkpoint)

    # model.load_state_dict(torch.load(cfg.model.checkpoint))

    # # Eval train and test
    # mean_train_dice, per_class_train_dice = Evaluate(model, train_loader, "training", cfg.model.num_classes)
    # formatted_per_class = [f"{x:.4f}" for x in per_class_train_dice]
    # print(f"Mean train dice: {mean_train_dice:.4f}. Mean train perclass dice: {per_class_train_dice}")

    # mean_eval_dice, per_class_eval_dice = Evaluate(model, test_loader, "testing", cfg.model.num_classes)
    # formatted_per_class = [f"{x:.4f}" for x in per_class_eval_dice]
    # print(f"Mean eval dice: {mean_eval_dice:.4f}. Mean eval perclass dice: {per_class_eval_dice}")

if __name__ == "__main__":
    main()