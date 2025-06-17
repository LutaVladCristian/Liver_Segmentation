import os
from os.path import exists
from glob import glob
import torch
import numpy as np
from monai.networks.nets import UNet, SegResNet, AttentionUnet, UNETR, VNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, TverskyLoss, DiceFocalLoss
from monai.utils import set_determinism

from utilities import train, get_initial_meta_data
from preprocess import preprocess_data

# We do the training on the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
print("Using device: ", device, "\n")

# The input paths for the prepared nifti files
nif_path = ['data_set_group_nif/nif_files_testing/images', 
            'data_set_group_nif/nif_files_testing/labels', 
            'data_set_group_nif/nif_files_training/images',
            'data_set_group_nif/nif_files_training/labels',]

# Save the metadata of the entire training set
pix_dim_avg = get_initial_meta_data(nif_path, 'training_volumes')
print(pix_dim_avg)

# Preprocess the data
data_in = preprocess_data(
    nif_path,
    batch_size=2,  # start conservative
    spatial_size=(128, 128, 32),
    pixdim=pix_dim_avg
)

# Initialize the models
model1 = UNETR(
    img_size=(128, 128, 32),
    in_channels=1,
    out_channels=2,
    feature_size=32,
    hidden_size=384,
    mlp_dim=1536,
    num_heads=6,
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0
).to(device)

model2 = AttentionUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2)
).to(device)

model3 = UNet(
  spatial_dims=3,
  in_channels=1,
  out_channels=2,
  channels=(16, 32, 64, 128),
  strides=(2, 2, 2),
  num_res_units=2,
  norm=Norm.BATCH,
).to(device)

model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    init_filters=8,
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    upsample_mode="deconv",
    dropout_prob=0.2
).to(device)

model5 = VNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    dropout_prob_down=0.5,
    dropout_prob_up=(0.5, 0.5),
    dropout_dim=3,
    bias=False
).to(device)

# Initialize the loss function and the optimizer
loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True, lambda_focal=0.5)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5, amsgrad=True)

model_dir = 'trained_models/post_training_SegResNet_128_128_32'
os.makedirs(model_dir, exist_ok=True)

# Uncomment only if you want to start training for a pretrained model
model.load_state_dict(torch.load(os.path.join(model_dir, 'best_metric_model.pth')))

# Train the model
if __name__ == '__main__':

    train(model=model,
          data_in=data_in,
          loss_function=loss_function,
          optimizer=optimizer,
          max_epochs=100,
          #start_epoch=28,
          model_dir=model_dir,
          test_interval=4,
          device=device
    )