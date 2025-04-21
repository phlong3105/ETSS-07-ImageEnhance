import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from net.model import AdaIR
from options import options as opt
from utils.dataset_utils import AdaIRTrainDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR


class AdaIRModel(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.net     = AdaIR(decoder=True)
        self.loss_fn = nn.L1Loss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss     = self.loss_fn(restored, clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=180)
        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="AdaIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")
    
    trainset            = AdaIRTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs = 1, save_top_k=-1)
    trainloader = DataLoader(
        trainset,
        batch_size  = opt.batch_size,
        pin_memory  = True,
        shuffle     = True,
        drop_last   = True,
        num_workers = opt.num_workers
    )
    
    model   = AdaIRModel()
    trainer = pl.Trainer(
        max_epochs  = opt.epochs,
        accelerator = "gpu",
        devices     = opt.num_gpus,
        strategy    = "ddp_find_unused_parameters_true",
        logger      = logger,
        callbacks   = [checkpoint_callback]
    )
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == "__main__":
    main()
