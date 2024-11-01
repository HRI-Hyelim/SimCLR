import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from pathlib import Path
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
import lightly.data as data
from torchvision.transforms import transforms
import pandas as pd
from data_aug.gaussian_blur import GaussianBlur
import time

torch.cuda.empty_cache()
batch_size = 128 # 256 doesn't work --> gives CUDA memory error
max_epoch = 100


exp_run = {'MTF_e100': 'MTF', 'MTF_e100_v1': 'MTF', 'MTF_e100_v2': 'MTF',
           'sum_e100': 'GAF_sum', 'sum_e100_v1': 'GAF_sum', 'sum_e100_v2': 'GAF_sum',
           'diff_e100': 'GAF_diff', 'diff_e100_v1': 'GAF_diff', 'diff_e100_v2': 'GAF_diff',
    'rec_e100': 'recurrence', 'rec_e100_v1': 'recurrence', 'rec_e100_v2': 'recurrence'}

class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 2048, 2048)

        # enable gather_distributed to gather features from all gpus
        # before calculating the loss
        self.criterion = NTXentLoss(gather_distributed=True)
        self.epoch_losses = []
        self.train_start_time = None
        self.train_end_time = None

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.tensor(self.trainer.callback_metrics["train_loss_ssl"]).clone().detach().mean().item()
        self.epoch_losses.append(avg_loss)
        print(f"Epoch {self.current_epoch} average loss: {avg_loss}")

    def on_train_start(self):
        self.train_start_time = time.time()

    
    def on_train_end(self):
        self.train_end_time = time.time()
        training_duration = self.train_end_time - self.train_start_time
        print(f"Training complete. Total training time: {training_duration: .2f} seconds")

        df = pd.DataFrame(self.epoch_losses, columns = ['avg_loss'])
        df.to_csv("epoch_losses_" + file_version + '.csv', index = False)
        print("training completed. Epoch losses saved to epoch_losses.csv")

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06, momentum = 0.9, weight_decay = 5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = 10)
        return [optim], [scheduler]




# transform = SimCLRTransform(input_size=32)

# dataset = torchvision.datasets.CIFAR10(
#     "datasets/cifar10", download=True, transform=transform
# )
# or create a dataset from a folder containing images or videos:
# size = 128
# s=1
# color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
# transform = transforms.Compose([    
#                                         transforms.RandomGrayscale(p=0.2),
#                                         transforms.RandomApply([color_jitter], p=0.8),
#                                         GaussianBlur(kernel_size=int(0.1 * size)),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                                     ])

for key, value in exp_run.items():

    file_version = key
    image_folder_path = Path.cwd() / value


    transform = SimCLRTransform(input_size=128) # why does input_size = 32 worked? doesn't make sense

    dataset = data.LightlyDataset(image_folder_path, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    # Train with DDP and use Synchronized Batch Norm for a more accurate batch norm
    # calculation. Distributed sampling is also enabled with replace_sampler_ddp=True.
    trainer = pl.Trainer(
        max_epochs=max_epoch,
        devices="auto",
        accelerator="gpu",
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=True,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
    )



    model = SimCLR()
    torch.cuda.empty_cache()
    trainer.fit(model=model, train_dataloaders=dataloader)
    torch.save(model.state_dict(), './' + file_version +'.pth')
    torch.cuda.empty_cache()

# model = YourModelClass()
# model.load_state_dict(torch.load('path_to_save_model/model.pth'))
# model.eval()  # Set the model to evaluation mode


# To do:
# Hyperparameter tuning: learning rate
