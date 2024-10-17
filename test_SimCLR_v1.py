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
from data_aug.gaussian_blur import GaussianBlur

class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 2048, 2048)

        # enable gather_distributed to gather features from all gpus
        # before calculating the loss
        self.criterion = NTXentLoss(gather_distributed=True)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim

torch.cuda.empty_cache()
model = SimCLR()

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



transform = SimCLRTransform(input_size=128) # why does input_size = 32 worked? doesn't make sense
image_folder_path = Path.cwd() / 'MTF'
dataset = data.LightlyDataset(image_folder_path, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

# Train with DDP and use Synchronized Batch Norm for a more accurate batch norm
# calculation. Distributed sampling is also enabled with replace_sampler_ddp=True.
trainer = pl.Trainer(
    max_epochs=20,
    devices="auto",
    accelerator="gpu",
    strategy="ddp",
    sync_batchnorm=True,
    use_distributed_sampler=True,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
)
torch.cuda.empty_cache()
trainer.fit(model=model, train_dataloaders=dataloader)
torch.save(model.state_dict(), './v2.pth')

# model = YourModelClass()
# model.load_state_dict(torch.load('path_to_save_model/model.pth'))
# model.eval()  # Set the model to evaluation mode


# To do:
# Hyperparameter tuning: learning rate
