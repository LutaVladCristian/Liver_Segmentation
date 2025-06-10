import os
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss

import torch
from utilities import train
from preprocess import preprocess_data


# The input paths for the prepared nifti files
nif_path = ['data_set_group_nif/nif_files_testing/images', 
            'data_set_group_nif/nif_files_testing/labels', 
            'data_set_group_nif/nif_files_training/images',
            'data_set_group_nif/nif_files_training/labels',]

# Preprocess the data
data_in = preprocess_data(nif_path, batch_size=1, spatial_size=(512, 512, 16))

# We do the training on the GPU
device = torch.device("cuda:0")

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
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

try:
    os.mkdir('post_training')
except:
    pass

# Train the model
if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, 10, 'post_training_best', test_interval=100, device=device)
