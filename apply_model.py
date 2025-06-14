from monai.utils import first
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import DataLoader, CacheDataset
from monai.transforms import(
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)

import torch
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import nibabel as nib


os.environ['KMP_DUPLICATE_LIB_OK'] = "1"

# Paths to the validation set and the model
validation_path = '../data_set_group_nif/nif_files_validation'
model_path = 'post_training_best'

path_validation_volumes = glob(os.path.join(validation_path, 'images/*'))
path_validation_labels = glob(os.path.join(validation_path, 'labels/*'))

validation_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(path_validation_volumes, path_validation_labels)]


# Transforms for the validation set
validation_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5,1.5,1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), 
        CropForegroundd(keys=["image", "label"], source_key='image'),
        Resized(keys=["image", "label"], spatial_size=[256,256,16]),   
        ToTensord(keys=["image", "label"]),
    ]
)

# Create the validation dataset
validation_ds = CacheDataset(data=validation_files, transform=validation_transforms)
validation_loader = DataLoader(validation_ds, batch_size=1)

# Set the device to cuda
device = torch.device('cuda:0')

# Load the model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

model.load_state_dict(torch.load(os.path.join(model_path, 'best_metric_model.pth')))

# Set the model to evaluation mode
model.eval()

# Create a folder for the mask volumes
try:
    os.mkdir('mask_volumes')
except:
    pass


# Infer for all the patients of the validation set
with torch.no_grad():

    for patient in validation_loader:
        volume = patient['image'].to(device)
        label = patient['label'].to(device)
        label = label != 0

        # Apply the model
        out = model(volume)
        out = torch.sigmoid(out)
        out = out > 0.5

        # Convert the tensor to numpy
        out = out.detach().cpu().numpy()
        out = out[0,1]

        label = label.detach().cpu().numpy()
        label = label[0,0]

        # Apply the mask
        out = out.float()
        for i in range(16):
            volume[:, :, i] = volume[:, :, i] * out[:, :, i]

        # Convert the volume from Numpy to NIfTI
        #nifti_image = nib.Nifti1Image(volume, affine = RAS_affine)

        # Save the NIfTI file
        #nib.save(nifti_image, 'volumes/inference.nii')
