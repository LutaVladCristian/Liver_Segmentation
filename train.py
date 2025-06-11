import os
from os.path import exists
from glob import glob
import torch
import numpy as np

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, TverskyLoss
from monai.utils import set_determinism
from monai.networks.utils import one_hot

from utilities import train
from preprocess import preprocess_data


# The input paths for the prepared nifti files
nif_path = ['data_set_group_nif/nif_files_testing/images', 
            'data_set_group_nif/nif_files_testing/labels', 
            'data_set_group_nif/nif_files_training/images',
            'data_set_group_nif/nif_files_training/labels',]

# Preprocess the data
data_in = preprocess_data(nif_path, batch_size=1, spatial_size=(16, 16, 2))

# We do the training on the GPU
device = torch.device("cuda:0")
print(device)

# Initialize the 3D U-Net model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

# Initialize the loss function and the optimizer
loss_function = TverskyLoss(
    to_onehot_y=True,
    softmax=True,
    alpha=0.7,  # penalize false negatives more (missing tumors)
    beta=0.3,
    include_background=True
)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5, amsgrad=True)


# Train the model
if __name__ == '__main__':
    model_dir = 'trained_models/post_training_Unet'
    os.makedirs(model_dir, exist_ok=True)

    train(model=model,
          data_in=data_in,
          num_classes=3,
          loss_function=loss_function,
          optimizer=optimizer,
          max_epochs=10,
          model_dir=model_dir,
          test_interval=1,
          device=device
          )