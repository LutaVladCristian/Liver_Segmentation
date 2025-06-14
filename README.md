# Liver Segmentation

### Introduction
Semantic segmentation is an image processing technique that consists of assigning labels to each pixel in an image. This technique is used in various fields, including medicine, to identify and isolate regions of interest in medical images.
In the field of medical imaging, semantic liver segmentation is an important issue in the diagnosis and treatment planning of liver diseases. The goal is to separate the liver region from medical images for a more detailed analysis and assessment of its health status. 3D reconstruction is a technique that uses data obtained from various sources, such as medical imaging, to create a detailed 3D image of the organ in question. Also, the 3D model gives doctors a more accurate perspective of the organ structure This technique is useful in many medical fields, such as planning surgical interventions, guiding interventional procedures, assessing, diagnosis, research, and development of new treatments.


### 3D-Unet Architecture
This architecture is an extension of the U-Net architecture for semantic segmentation in volumetric data. We can observe that the structure is also based on an encoder and decoder, the only difference being the adding an extra dimension to the input and output data.

![WhatsApp Image 2024-01-27 at 18 12 35_8f66c8a5](https://github.com/LutaVladCristian/Liver_Segmentation/assets/62925188/7820e30c-b2d4-4a8c-8f9c-9f1c90a577dc)

The architecture of the 3D U-Net consists of an analysis path and a synthesis path, each with four resolution steps. In the encoder path, each layer consists of two convolutions of size 3×3×3, followed by a ReLU layer and a max pooling layer of size 2×2×2, with stride 2 for each dimension. In the decoder path, each layer consists of an up-convolution operation of size 2×2×2, with stride 2 in each dimension, followed by two convolutions of size 3×3×3, each followed by a ReLU layer. Skip-connections are used to ensure connectivity between equal resolution layers in the encoder path and the decoder path. These connections allow the transfer of essential high-resolution features from the encoder path to the decoder path. In the last layer, a convolution of size 1×1×1 is applied to reduce the number of output channels to the number of desired labels, which in this case is 2. The total number of architecture parameters is 19,069,955. In terms of batch normalization (BN), in this architecture, BN is inserted before each ReLU layer.


### Implementation and workflow
In the following, I will make a presentation on personal implementation. At the same time, the code is composed of six separate modules, which can be categorized according to their role in three key stages of research. The figure below illustrates the data flow.

![WhatsApp Image 2024-01-27 at 18 32 09_9d3d81f7](https://github.com/LutaVladCristian/Liver_Segmentation/assets/62925188/3bb8350d-37b3-4dcf-8795-ceeb4f2770dd)

**Data preparation** involves the preparation and pre-processing of the dataset prior to the model training phase. The dataset contains volumetric data obtained from CT analysis, as well as the related labels for 130 patients. I divided the 130 volumes in the following way: 70% training data, 15% test data, and 15% data from the validation data.

![WhatsApp Image 2024-01-27 at 18 20 37_dd0cb86f](https://github.com/LutaVladCristian/Liver_Segmentation/assets/62925188/ce6bcb63-43fd-43f1-933e-31f96bee591c)

I also noticed that patients have a different number of cross-sections, so I made the decision to divide each volume into sub-volumes of 16 sections each using the **prepare.py** module. I chose this number of sections due to the fact that if I grouped the subvolumes into too large groups, I risked losing useful information, for example: for a patient who has 310 sections contained in the volume and I want to divide it into groups of 64 sections each, in the end, I lose a total of 54 sections from the initial volume, which can cause loss of liver volume.

At the same time, the volumes in the dataset include the body part immediately below the cervical area to the beginning of the groin area, so I took the decision to eliminate groups that contained virtually no liver-related labels within the sections. This resulted in a significant decrease in V-RAM requirements. To manipulate the data and organize it into directories I used the modules included in Python: os, globe, and shutil, and to check if they contained the volume corresponding to the liver I used NiBabel to convert Niffti files into NumPy vectors.

**Preprocess.py** is designed to load previously prepared and filtered data into the V-RAM memory of the GPU to significantly decrease the time required for training. This module is also intended for applying transformations to regularize the data set, but also for augmenting the training data, the most important ones being:
- **Spacingd**: is intended to adjust the spatial resolution of the data volumes
- **ScaleIntensityRanged**: normalizes the intensities of the pixels in the image

![WhatsApp Image 2024-01-27 at 18 38 28_daf8dea9](https://github.com/LutaVladCristian/Liver_Segmentation/assets/62925188/7e0cdced-88d5-42dd-a4f3-af40a56e0a29)

- **RandAffined**: is used to augment the training data and applies random value translations to the image
- **RandRotated**: applies rotations of random values
- **RandGaussianNoised**: applies noise to the image and is applied exclusively to the training data
- **Resized**: is intended to resize images. In my case, I used a resolution of 256x256
- **ToTensord**: aims to transform images into tensors

**Utilities.py** contains three functions: show_patient for viewing sections and their labels, dice_metric for calculating the Dice coefficient used as a metric in the training process, and the train function representing the training loop. The train function has a number of parameters: the model we are training, the training data, the optimizer, the number of epochs, the directory where we want to save the training checkpoints, the range of epochs for which we want to test the model and the device on which we want to perform the training (GPU or CPU).

**Train.py** is the module for training. This contains the call to the training and test data preprocessing function, the initialization of the model, the cost function, the Adam optimizer, and the call of the train function. The Adam optimizer offers a balance between computational efficiency, learning rate adaptability, and training stabilization in deep neural networks, making it a perfect choice for conducting my research.

**Apply_model.py** aims to receive as input a new volume that has not been included in the training set before. Based on the model we will obtain its predictions in the form of binary masks corresponding to each section of the volume (0 for background pixels and 1 for pixels classified as part of the target organ). We then apply the model predictions on each section of the data volume by a multiplication operation of the images with the corresponding mask (logical AND between masks and corresponding slices). The resulting volume will be converted using NiBabel to a Niffti file and forwarded for volumetric visualization.

![WhatsApp Image 2024-01-27 at 18 41 36_d2457bf8](https://github.com/LutaVladCristian/Liver_Segmentation/assets/62925188/46b5ef37-e60a-43f0-8651-55d6b3a7b918)

**3D-vizualization.py** is used to volumetrically visualize the result of the previous step presented using the Marching Cubes algorithm (IsosurfaceBrowser from Vedo) or Ray Casting (RayCastPlotter from Vedo). It can be seen that the Ray Casting algorithm reveals the tumor tissues when set in max projection mode. However, we can see that in both cases, the 3D model has a sharp rather than smooth appearance due to the losses caused by segmentation.

![WhatsApp Image 2024-01-27 at 18 43 50_c0e40bf6](https://github.com/LutaVladCristian/Liver_Segmentation/assets/62925188/637930a0-84f5-44b0-86de-23a295f549e0)


### Conclusions
I observe that the reconstruction of the 3D model from the slices obtained after the multiplication of the initial slices and their corresponding masks suffers shape loss. This may indicate inaccuracy or distortion of the original structure following the segmentation and reconstruction process. Also, from the 3D reconstruction of the organ we can calculate certain constructive characteristics, such as volume, or observe the shape, and color.

At the same time, because we are dealing with supervised learning, we can train the network on any organ contained in the volumetric data as long as we have the corresponding labels. If we would like to extend the study from organ segmentation to tumor segmentation or pathology extraction, we need to change the approach, in that both the network architecture and the cost function are not suitable for such small target segmentation tasks compared to the background. For example, we can use what is called "weighted cross-entropy" as a cost function, which has the ability to adapt to unbalanced datasets (e.g. small tumors in a large volume).




