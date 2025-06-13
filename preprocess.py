import os
from glob import glob
from monai.data import Dataset, CacheDataset,DataLoader
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandAffined,
    RandRotated,
    RandGaussianNoised,
    CropForegroundd,
    Orientationd,
    Resized,
    ToTensord,
    Spacingd,
    EnsureTyped,
)


os.environ['KMP_DUPLICATE_LIB_OK']='True'

def preprocess_data(data_path, batch_size=8, spatial_size=(256, 256, 16)):
    
    set_determinism(seed=0)
    
    # Create the dataset
    test_data = sorted(glob(data_path[0] + f'/*'))
    test_labels = sorted(glob(data_path[1] + f'/*'))

    train_data = sorted(glob(data_path[2] + f'/*'))
    train_labels = sorted(glob(data_path[3] + f'/*'))

    train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_data, train_labels)]
    test_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_data, test_labels)]

    # Transforms for the training with data augmentation
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),  # Load the images
        EnsureChannelFirstd(keys=["image", "label"]),  # Ensure the channel is the first dimension of the image
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),  # Resample the images
        Orientationd(keys=["image", "label"], axcodes="RAS"),  # Change the orientation of the image
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        # Change the contrast of the image and gives the image pixels,
        # values between 0 and 1
        CropForegroundd(keys=["image", "label"], source_key="image"),  # Crop foreground of the image
        RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10),  # Randomly shift the image
        RandRotated(keys=['image', 'label'], prob=0.5, range_x=10.0),  # Randomly rotate the image
        RandGaussianNoised(keys='image', prob=0.5),  # Add random noise to the image
        Resized(keys=["image", "label"], spatial_size=spatial_size),  # Resize the image
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),  # Convert the images to tensors
    ])

    # Transforms for the testing
    test_transforms = Compose(# Compose transforms together
        [
            LoadImaged(keys=["image", "label"]),  # Load the images
            EnsureChannelFirstd(keys=["image", "label"]),  # Ensure the channel is the first dimension of the image
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            # Resample the images
            Orientationd(keys=["image", "label"], axcodes="RAS"),  # Change the orientation of the image
            ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
            # Change the contrast of the image and gives the image pixels,
            # values between 0 and 1
            CropForegroundd(keys=["image", "label"], source_key="image"),  # Crop foreground of the image
            Resized(keys=["image", "label"], spatial_size=spatial_size),  # Resize the image
            EnsureTyped(keys=["image", "label"]),
            ToTensord(keys=["image", "label"]),  # Convert the images to tensors
        ]
    )

    # Create the datasets
    train_ds = CacheDataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size)

    test_ds = CacheDataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, test_loader