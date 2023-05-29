"""
Created on Thu Apr  6 14:29:08 2023

@author: vlad_cristian.luta
"""

import os
from glob import glob
import torch
from monai.data import Dataset, DataLoader
from monai.utils import first
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
    Resized,
    ToTensord,
    Spacingd,
)
import utilities

# The input paths for the prepared nifti files
nif_path = ['data_set_group_nif/nif_files_testing/images', 
            'data_set_group_nif/nif_files_testing/labels', 
            'data_set_group_nif/nif_files_training/images',
            'data_set_group_nif/nif_files_training/labels', 
            'data_set_group_nif/nif_files_validation/images',
            'data_set_group_nif/nif_files_validation/labels']

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def preprocess_data(data_path, batch_size=1, pixdim=(1.5, 1.5, 2.0), a_min=-200, a_max=200, spatial_size=(256, 256, 74)):
    # Create the dataset
    test_data = sorted(glob(data_path[0] + f'/*'))
    test_labels = sorted(glob(data_path[1] + f'/*'))

    train_data = sorted(glob(data_path[2] + f'/*'))
    train_labels = sorted(glob(data_path[3] + f'/*'))

    val_data = sorted(glob(data_path[4] + f'/*'))
    val_labels = sorted(glob(data_path[5] + f'/*'))

    train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_data, train_labels)]
    val_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_data, val_labels)]
    test_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_data, test_labels)]

    # Create the transforms
    original_transforms = Compose(# Compose transforms together
        [
            LoadImaged(keys=["image", "label"]), # Load the images
            EnsureChannelFirstd(keys=["image", "label"]), # Add a channel to the images that will control the batch number
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")), # Resample the images
            ToTensord(keys=["image", "label"]), # Convert the images to tensors
        ]
    )

    train_transforms = Compose(# Compose transforms together
        [
            LoadImaged(keys=["image", "label"]), # Load the images
            EnsureChannelFirstd(keys=["image", "label"]), # Add a channel to the images that will control the batch number
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")), # Resample the images
            #Orientationd(keys=["image", "label"], axcodes="RAS"), # Change the orientation of the image
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), # Change the contrast of the image and gives the image pixels, 
                                                                                                                    #values between 0 and 1
            CropForegroundd(keys=["image", "label"], source_key="image"), # Crop foreground of the image
            #Orientationd(keys=["image", "label"], axcodes="RAS"), # Change the orientation of the image
            Resized(keys=["image", "label"], spatial_size=spatial_size), # Resize the image
            ToTensord(keys=["image", "label"]), # Convert the images to tensors
        ]
    )

    val_transforms = Compose(# Compose transforms together
        [
            LoadImaged(keys=["image", "label"]), # Load the images
            EnsureChannelFirstd(keys=["image", "label"]), # Add a channel to the images that will control the batch number
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")), # Resample the images
            #Orientationd(keys=["image", "label"], axcodes="RAS"), # Change the orientation of the image
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), # Change the contrast of the image and gives the image pixels, 
                                                                                                                    #values between 0 and 1
            CropForegroundd(keys=["image", "label"], source_key="image"), # Crop foreground of the image
            #Orientationd(keys=["image", "label"], axcodes="RAS"), # Change the orientation of the image
            Resized(keys=["image", "label"], spatial_size=spatial_size), # Resize the image
            ToTensord(keys=["image", "label"]), # Convert the images to tensors
        ]
    )

    test_transforms = Compose(# Compose transforms together
        [
            LoadImaged(keys=["image", "label"]), # Load the images
            EnsureChannelFirstd(keys=["image", "label"]), # Add a channel to the images that will control the batch number
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")), # Resample the images
            #Orientationd(keys=["image", "label"], axcodes="RAS"), # Change the orientation of the image
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), # Change the contrast of the image and gives the image pixels, 
                                                                                                                    #values between 0 and 1
            CropForegroundd(keys=["image", "label"], source_key="image"), # Crop foreground of the image
            #Orientationd(keys=["image", "label"], axcodes="RAS"), # Change the orientation of the image
            Resized(keys=["image", "label"], spatial_size=spatial_size), # Resize the image
            ToTensord(keys=["image", "label"]), # Convert the images to tensors
        ]
    )

    # Create the datasets
    original_ds = Dataset(data=train_files, transform=original_transforms)
    original_loader = DataLoader(original_ds, batch_size=batch_size)

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size)

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Show the data
    utilities.show_patient(train_loader, SLICE_NUMBER=57, show_original=True, original_data=original_loader)
    #utilities.show_patient(test_loader, SLICE_NUMBER=40)
    #utilities.show_patient(val_loader, SLICE_NUMBER=40)

    return train_loader, val_loader, test_loader





preprocess_data(nif_path, batch_size=1)