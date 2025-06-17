from monai.utils import first
from monai.networks.nets import UNet, SegResNet, AttentionUnet, UNETR, VNet
from monai.networks.layers import Norm
from monai.data import DataLoader, CacheDataset
from monai.transforms import(
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    EnsureTyped,
    ToTensord,
)

import torch
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import nibabel as nib

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Set the device to cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
print("Using device: ", device, "\n")

# Paths to the validation set and the model
validation_path = '../data_set_group_nif/nif_files_validation'

path_validation_volumes = glob(os.path.join(validation_path, 'images/*'))
path_validation_labels = glob(os.path.join(validation_path, 'labels/*'))

validation_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(path_validation_volumes, path_validation_labels)]

# Transforms for the validation set
validation_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(0.7871384, 0.7871384, 1.2131842), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=250,b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key='image', allow_missing_keys=True),
        Resized(keys=["image", "label"], spatial_size=(128, 128, 32)),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ]
)

# Create the validation dataset
validation_ds = CacheDataset(data=validation_files, transform=validation_transforms)
validation_loader = DataLoader(validation_ds, batch_size=1)

# Load the models that we want to evaluate
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

model4 = SegResNet(
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

model_path = 'trained_models/post_training_AttentionUNet_128_128_32'
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
