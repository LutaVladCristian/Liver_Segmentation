"""
Created on Thu Apr  6 14:29:08 2023

@author: vlad_cristian.luta
"""

import os
from glob import glob
import shutil
import dicom2nifti
import nibabel as nib
import numpy as np


# The input paths for the dicom files
in_path = ['data_set/dicom_files_testing/images', 
            'data_set/dicom_files_testing/labels', 
            'data_set/dicom_files_training/images',
            'data_set/dicom_files_training/labels',
            'data_set/dicom_files_validation/images',
            'data_set/dicom_files_validation/labels']

# The output paths for the dicom files split into equal groups
group_path = ['data_set_group/dicom_files_testing/images', 
            'data_set_group/dicom_files_testing/labels', 
            'data_set_group/dicom_files_training/images',
            'data_set_group/dicom_files_training/labels',
            'data_set_group/dicom_files_validation/images',
            'data_set_group/dicom_files_validation/labels']

# The output paths for the nifti files
nif_path = ['data_set_group_nif/nif_files_testing/images', 
            'data_set_group_nif/nif_files_testing/labels', 
            'data_set_group_nif/nif_files_training/images',
            'data_set_group_nif/nif_files_training/labels',
            'data_set_group_nif/nif_files_validation/images',
            'data_set_group_nif/nif_files_validation/labels']


# Function to group equaly the dicom files
def divide_groups(data_in, data_out, minNr_slices=64):

    # Delete the folders for the groups
    try:
        shutil.rmtree('data_set_group')
    except:
        pass

    # Create the folders for the groups
    try:
        os.mkdir('data_set_group')

        os.mkdir('data_set_group/dicom_files_testing')
        os.mkdir(data_out[0])
        os.mkdir(data_out[1])

        os.mkdir('data_set_group/dicom_files_training')
        os.mkdir(data_out[2])
        os.mkdir(data_out[3])

        os.mkdir('data_set_group/dicom_files_validation')
        os.mkdir(data_out[4])
        os.mkdir(data_out[5])

    except:
        pass

    # Divide the dicom files into groups
    for k in range(len(data_in)):
        # Testing data
        for patient in glob(data_in[k] + '/*'):
            patient = patient.split('\\')[1]
            #print(patient)

            noSlices = len(glob(data_in[k] + f'/{patient}/*'))
            #print(noSlices)

            ratio = noSlices // minNr_slices
            #print(ratio)

            for i in range(ratio):
                try:
                    os.mkdir(data_out[k] + f'/{patient}_{i}')
                    j = 0
                    #print(f'/{patient}_{i}')
                    for slice in glob(data_in[k] + f'/{patient}/*'):
                        if j >= minNr_slices:
                            break
                        slice = slice.split('\\')[1]
                        #print(slice)
                        shutil.move(data_in[k] + f'/{patient}/{slice}', data_out[k] + f'/{patient}_{i}')
                        j += 1
                except Exception as e:
                    print(e)
                    pass

    # Delete the folders for the groups
    try:
        shutil.rmtree('data_set')
    except:
        pass
divide_groups(in_path, group_path, 16)


# Function to convert the dicom files back to nifti
def convert_nif(data_in, data_out):

    # Create the folders for the nifti files
    try:
        os.mkdir('data_set_group_nif')

        os.mkdir('data_set_group_nif/nif_files_testing')
        os.mkdir(data_out[0])
        os.mkdir(data_out[1])

        os.mkdir('data_set_group_nif/nif_files_training')
        os.mkdir(data_out[2])
        os.mkdir(data_out[3])

        os.mkdir('data_set_group_nif/nif_files_validation')
        os.mkdir(data_out[4])
        os.mkdir(data_out[5])

    except:
        pass

    for i in range(len(data_out)):
        list_images = glob(data_in[i] + '/*')
        for patient in list_images:
            patient = patient.split('\\')[1]
            dicom2nifti.dicom_series_to_nifti(data_in[i] + f'/{patient}', data_out[i] + f'/{patient}.nii.gz')
            print(patient)
    
    # Delete the folders for the groups
    try:
        shutil.rmtree('data_set_group')
    except:
        pass
convert_nif(group_path, nif_path)


# Function to remove empty groups
def remove_empty_groups(data_in):
    for i in range(1, len(data_in), 2):
        list_patients = glob(data_in[i] + '/*')
        for patient in list_patients:
            nifti_file = nib.load(patient)
            fdata = nifti_file.get_fdata()

            # Check if the nifti file has only one value, then it means it does not have a label
            if len(np.unique(fdata)) == 1:
                print(patient)
                try:
                    os.remove(patient)
                except Exception as e:
                    print(e)
                    pass
                patient = patient.split('\\')[1]
                print(os.path.join(data_in[i-1], patient))
                try:
                    os.remove(os.path.join(data_in[i-1], patient))
                except Exception as e:
                    print(e)
                    pass
remove_empty_groups(nif_path)