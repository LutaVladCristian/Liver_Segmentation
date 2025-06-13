import os
from os.path import exists
from glob import glob
import torch
import numpy as np

from monai.networks.nets import UNet, AttentionUnet, UNETR, SwinUNETR, VNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, TverskyLoss, DiceFocalLoss
from monai.utils import set_determinism

from utilities import train
from preprocess import preprocess_data


# The input paths for the prepared nifti files
nif_path = ['data_set_group_nif/nif_files_testing/images', 
            'data_set_group_nif/nif_files_testing/labels', 
            'data_set_group_nif/nif_files_training/images',
            'data_set_group_nif/nif_files_training/labels',]

# Preprocess the data
data_in = preprocess_data(nif_path, batch_size=8, spatial_size=(96, 96, 16))

# We do the training on the GPU
device = torch.device("cuda:0")
print(device)

# Initialize the model
model = UNETR(
    in_channels=1,
    out_channels=2,
    img_size=(96, 96, 16),
    feature_size=16,            # Base channel size; you can try 32 if you have more memory
    hidden_size=768,            # Transformer hidden dimension
    mlp_dim=3072,               # Feedforward dim in Transformer
    num_heads=12,               # Attention heads
    norm_name='instance',       # Normalization layer
    res_block=True,             # Use residual blocks in conv path
    dropout_rate=0.0            # You can adjust this
).to(device)

# Initialize the loss function and the optimizer
loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True, lambda_focal=0.5)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5, amsgrad=True)


# Train the model
if __name__ == '__main__':
    model_dir = 'trained_models/post_training_UNETR'
    os.makedirs(model_dir, exist_ok=True)

    train(model=model,
          data_in=data_in,
          loss_function=loss_function,
          optimizer=optimizer,
          max_epochs=100,
          model_dir=model_dir,
          test_interval=4,
          device=device
    )