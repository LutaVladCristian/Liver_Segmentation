# Liver Segmentation

### Introduction
Semantic segmentation is an image processing technique that consists of 
assigning labels to each pixel in an image. This technique 
is used in various fields, including medicine, to identify and isolate regions of 
of interest in medical images.
In the field of medical imaging, semantic liver segmentation is a 
important issue in the diagnosis and treatment planning of liver diseases. 
liver diseases. The goal is to separate the liver region from medical images so as to 
more detailed analysis and assessment of its health status can be performed.
3D reconstruction is a technique used in the medical field to create a 
three-dimensional representation of a specific organ in the human body. This technique 
uses data obtained from various sources, such as medical imaging, to create a 
detailed 3D image of the organ in question.
3D reconstruction of a target organ also gives doctors a more accurate perspective of the organ. 
structure and its relationship to the organs and tissues of the body. 
adjacent tissues. This technique is useful in many medical fields, such as planning 
surgical interventions, guiding interventional procedures, assessing and 
diagnosis, and research and development of new therapies and treatments.


### 3D-Unet Architecture
This architecture is an extension of the U-Net architecture presented above 
for semantic processing and segmentation in volumetric data. One can observe the principle 
of structuring it also based on an encoder and decoder, the only difference being only the 
adding an extra dimension to the input and output data.

![WhatsApp Image 2024-01-27 at 18 12 35_8f66c8a5](https://github.com/LutaVladCristian/Liver_Segmentation/assets/62925188/7820e30c-b2d4-4a8c-8f9c-9f1c90a577dc)

















The architecture of the 3D U-Net has a similar structure to the standard U-Net, consisting of an analysis path and a synthesis path, each with four resolution steps. In the analysis path, each layer consists of two convolutions of size 3×3×3, followed by a ReLU layer and a max pooling stripe of size 2×2×2, with stride 2 for each dimension. In the synthesis path, each layer consists of an up-convolution operation of size 2×2×2, with stride 2 in each dimension, followed by two convolutions of size 3×3×3, each followed by a ReLU layer. Skip-connections are used to ensure connectivity between equal resolution layers in the analysis path and the synthesis path. These connections allow the transfer of essential high-resolution features from the analysis path to the synthesis path. In the last layer, a convolution of size 1×1×1 is applied to reduce the number of output channels to the number of desired labels, which in this case is 3. The total number of architecture parameters is 19,069,955. In terms of batch normalization (BN), in this architecture BN is inserted before each ReLU layer.


### Implementation and workflow
In the following, I will make a presentation on personal implementation and 
application of the knowledge presented in chapter two. At the same time, the code incorporates 
all the technologies presented above and is composed of six separate modules, which can be 
according to their role in three key stages of research. The figure below illustrates 
the dataflow.
Data preparation involves the preparation and pre-processing of the dataset prior to 
model training phase. The dataset contains volumetric data obtained from 
CT analysis, as well as the related labels for 130 patients. We divided the 130 
volumes in the following way: 70% training data, 15% test data and 15% data from the 
validation data.

