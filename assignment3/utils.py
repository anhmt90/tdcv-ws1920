import numpy as np
import data_generator
import cv2
import torch


def convert_to_numpy_array(tensor1, tensor2):
    if torch.is_tensor(tensor1):
        tensor1 = tensor1.numpy()
    if torch.is_tensor(tensor2):
        tensor2 = tensor2.numpy()
    return tensor1, tensor2

def knn_to_dbdataset(test_descriptors, db_descriptors):
    test_descriptors, db_descriptors = convert_to_numpy_array(test_descriptors, db_descriptors)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.match(test_descriptors, db_descriptors)
    return matches

def compute_angle(quaternion1, quaternion2):
    assert quaternion1.shape[0] == quaternion2.shape[0] == 4
    quaternion1, quaternion2 = convert_to_numpy_array(quaternion1, quaternion2)
    dot_res = np.clip(quaternion1 @ quaternion2, -1, 1)
    return 2 * np.rad2deg(np.arccos(np.abs(dot_res)).item())






