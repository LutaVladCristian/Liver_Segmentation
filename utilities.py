"""
Created on Thu Apr  6 14:29:08 2023

@author: vlad_cristian.luta
"""

from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm


def show_patient(data, SLICE_NUMBER=1, show_original=False, original_data=None):
    view_patient = first(data)

    if show_original:
        original_patient = first(original_data)
        plt.figure("Visualization", (12, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"original image {SLICE_NUMBER}")
        plt.imshow(original_patient["image"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 3, 2)
        plt.title(f"image {SLICE_NUMBER}")
        plt.imshow(view_patient["image"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 3, 3)
        plt.title(f"label {SLICE_NUMBER}")
        plt.imshow(view_patient["label"][0, 0, :, :, SLICE_NUMBER])
        plt.show()
    
    else:
        plt.figure("Visualization", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"image {SLICE_NUMBER}")
        plt.imshow(view_patient["image"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"label {SLICE_NUMBER}")
        plt.imshow(view_patient["label"][0, 0, :, :, SLICE_NUMBER])
        plt.show()

