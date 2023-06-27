from lungmask import mask 
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from tqdm import tqdm
from lungmask import utils
import sys
import pydicom
import logging
import time


def read_dicoms(path, original=False, primary=False):
    allfnames = []
    for dir, _, fnames in os.walk(path):
        [allfnames.append(os.path.join(dir, fname)) for fname in fnames]

    dcm_header_info = []
    unique_set = []  # need this because too often there are duplicates of dicom files with different names
    i = 0
    for fname in tqdm(allfnames):
        filename_ = os.path.splitext(os.path.split(fname)[1])
        i += 1
        if filename_[0] != 'DICOMDIR':
            try:
                dicom_header = pydicom.dcmread(fname, defer_size=100, stop_before_pixels=True, force=True)
                if dicom_header is not None:
                    if 'ImageType' in dicom_header:
                        if primary:
                            is_primary = all([x in dicom_header.ImageType for x in ['PRIMARY']])
                        else:
                            is_primary = True

                        if original:
                            is_original = all([x in dicom_header.ImageType for x in ['ORIGINAL']])
                        else:
                            is_original = True

                        # if 'ConvolutionKernel' in dicom_header:
                        #     ck = dicom_header.ConvolutionKernel
                        # else:
                        #     ck = 'unknown'
                        if is_primary and is_original and 'LOCALIZER' not in dicom_header.ImageType:
                            h_info_wo_name = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID,
                                              dicom_header.ImagePositionPatient]
                            h_info = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID, fname,
                                      dicom_header.ImagePositionPatient]
                            if h_info_wo_name not in unique_set:
                                unique_set.append(h_info_wo_name)
                                dcm_header_info.append(h_info)
                                # kvp = None
                                # if 'KVP' in dicom_header:
                                #     kvp = dicom_header.KVP
                                # dcm_parameters.append([ck, kvp,dicom_header.SliceThickness])
            except:
                logging.error("Unexpected error:", sys.exc_info()[0])
                logging.warning("Doesn't seem to be DICOM, will be skipped: ", fname)
    conc = [x[1] for x in dcm_header_info]
    sidx = np.argsort(conc)
    conc = np.asarray(conc)[sidx]
    dcm_header_info = np.asarray(dcm_header_info)[sidx]
    # dcm_parameters = np.asarray(dcm_parameters)[sidx]
    vol_unique = np.unique(conc, return_index=1, return_inverse=1)  # unique volumes
    n_vol = len(vol_unique[1])
    logging.info('There are ' + str(n_vol) + ' volumes in the study')

    for i in range(len(vol_unique[1])):
        curr_vol = i
        info_idxs = np.where(vol_unique[2] == curr_vol)[0]
        vol_files = dcm_header_info[info_idxs, 2]
        positions = np.asarray([np.asarray(x[2]) for x in dcm_header_info[info_idxs, 3]])
        slicesort_idx = np.argsort(positions)
        vol_files = vol_files[slicesort_idx]

    return vol_files


def get_pixels(path):
    image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)[0, :, :]

    return image

orignal_path = r'F:\SliceSelection\Top10\Validation'
masked_save_path = r'F:\SliceSelection\Top10\MaskVal'

for patient in os.listdir(orignal_path):
    patient_folder = os.path.join(orignal_path, patient)
    pos_path = os.path.join(patient_folder, '1')
    neg_path = os.path.join(patient_folder, '0')

    save_patient_folder = os.path.join(masked_save_path, patient)
    save_pos_path = os.path.join(save_patient_folder, '1')
    save_neg_path = os.path.join(save_patient_folder, '0')

    if not os.path.exists(save_pos_path):
        os.makedirs(save_pos_path)
    if not os.path.exists(save_neg_path):
        os.makedirs(save_neg_path)

    print('precessing ', pos_path)
    if len(os.listdir(pos_path)):
        dcms = read_dicoms(pos_path)
        input_image = utils.get_input_image(pos_path)
        segmentation = mask.apply(input_image)

        for index, dcm in enumerate(tqdm(dcms)):
            image_name = dcm.split('\\')[-1][:-4]
            npy_save_name = image_name + '.npy'
            raw_image_arr = get_pixels(dcm)
            mask_array = segmentation[index]
            mask_array[mask_array > 0] = 1
            raw_image_arr[np.where(mask_array == 0)] = -1024
            save_npy_path = os.path.join(save_pos_path, npy_save_name)
            np.save(save_npy_path, raw_image_arr)

    print('precessing ', neg_path)
    if len(os.listdir(neg_path)):
        dcms = read_dicoms(neg_path)
        input_image = utils.get_input_image(neg_path)
        segmentation = mask.apply(input_image)
        for index, dcm in enumerate(tqdm(dcms)):
            image_name = dcm.split('\\')[-1][:-4]
            npy_save_name = image_name + '.npy'
            raw_image_arr = get_pixels(dcm)
            mask_array = segmentation[index]
            mask_array[mask_array > 0] = 1
            raw_image_arr[np.where(mask_array == 0)] = -1024
            save_npy_path = os.path.join(save_neg_path, npy_save_name)
            np.save(save_npy_path, raw_image_arr)

