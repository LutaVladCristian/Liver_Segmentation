"""
Created on Thu Apr  6 14:29:08 2023

@author: vlad_cristian.luta
"""

from glob import glob

# Test the number of slices in a group
def test_group():
    # The minimum number of slices in a group
    minNr_slices = 1000000

    # The maximum number of slices in a group
    maxNr_slices = 0

    # The input paths for the dicom files
    data_in = ['data_set_group/dicom_files_testing/images', 
                'data_set_group/dicom_files_testing/labels', 
                'data_set_group/dicom_files_training/images',
                'data_set_group/dicom_files_training/labels', 
                'data_set_group/dicom_files_validation/images',
                'data_set_group/dicom_files_validation/labels']


    for path in data_in:
        for patient in glob(path + f'/*'):
            #print (patient)
            noSlices = len(glob(patient + f'/*'))
            #print(noSlices)
            if noSlices < minNr_slices:
                minNr_slices = noSlices
            if noSlices > maxNr_slices:
                maxNr_slices = noSlices
    return minNr_slices, maxNr_slices

assert(test_group() == (74, 74))
minNr_slices, maxNr_slices = test_group()
print(minNr_slices, maxNr_slices)