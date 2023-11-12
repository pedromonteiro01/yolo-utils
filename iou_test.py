# Author: Pedro Monteiro
# Date: November 2023
# Computer Science Engineering MSc
# Aveiro University

import os
import torch
import pytest
from iou2 import load_tensor_from_file, get_sample_data, yolo_to_corner, compute_average_iou, main
from unittest.mock import patch, Mock

@pytest.fixture
def mock_data():
    detect_folder = 'mock_detect_folder'
    valid_folder = 'mock_valid_folder'
    os.makedirs(detect_folder, exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)

    # create mock detection file simulating YOLO's output with multiple objects
    with open(os.path.join(detect_folder, 'mock_detect.txt'), 'w') as f:
        f.write('0 0.5 0.5 0.5 0.5 0.9\n')  # Object 1
        f.write('1 0.7 0.7 0.2 0.2 0.8\n')  # Object 2

    # create mock validation label file with multiple objects
    with open(os.path.join(valid_folder, 'mock_valid.txt'), 'w') as f:
        f.write('0 0.5 0.5 0.4 0.4\n')  # Object 1
        f.write('1 0.7 0.7 0.2 0.2\n')  # Object 2

    # create mock detection file with 3 objects
    with open(os.path.join(detect_folder, 'mock_detect_3objs.txt'), 'w') as f:
        f.write('0 0.5 0.5 0.5 0.5 0.9\n')  # Object 1
        f.write('1 0.7 0.7 0.2 0.2 0.8\n')  # Object 2
        f.write('2 0.3 0.3 0.2 0.2 0.7\n')  # Object 3

    # create mock validation label file with 3 objects
    with open(os.path.join(valid_folder, 'mock_valid_3objs.txt'), 'w') as f:
        f.write('0 0.5 0.5 0.4 0.4\n')  # Object 1
        f.write('1 0.7 0.7 0.2 0.2\n')  # Object 2
        f.write('2 0.3 0.3 0.2 0.2\n')  # Object 3

    # create mock 'exp' directory and detection file
    exp_folder = os.path.join(detect_folder, 'batch2_epoch10')
    os.makedirs(exp_folder, exist_ok=True)
    labels_folder = os.path.join(exp_folder, 'labels')
    os.makedirs(labels_folder, exist_ok=True)
    with open(os.path.join(labels_folder, 'mock_detect_exp.txt'), 'w') as f:
        f.write('0 0.5 0.5 0.5 0.5 0.9\n')  # Object 1

    yield detect_folder, valid_folder

    # cleanup mock directories after testing
    os.remove(os.path.join(labels_folder, 'mock_detect_exp.txt'))
    os.rmdir(labels_folder)
    os.rmdir(exp_folder)
    os.remove(os.path.join(detect_folder, 'mock_detect.txt'))
    os.remove(os.path.join(valid_folder, 'mock_valid.txt'))
    os.remove(os.path.join(detect_folder, 'mock_detect_3objs.txt'))
    os.remove(os.path.join(valid_folder, 'mock_valid_3objs.txt'))
    os.rmdir(detect_folder)
    os.rmdir(valid_folder)
def test_load_tensor_from_file(mock_data):
    detect_folder, _ = mock_data
    tensor = load_tensor_from_file(os.path.join(detect_folder, 'mock_detect.txt'))
    expected_tensor = torch.tensor([[0, 0.5, 0.5, 0.5, 0.5, 0.9],  # object 1
                                    [1, 0.7, 0.7, 0.2, 0.2, 0.8]])  # object 2
    assert torch.equal(tensor, expected_tensor)

def test_get_sample_data(mock_data):
    detect_folder, _ = mock_data
    detections = get_sample_data(detect_folder)
    assert detections[0].shape[0] == 2  # check the number of rows in the first tensor

def test_yolo_to_corner():
    box = torch.tensor([0.5, 0.5, 0.5, 0.5])
    corners = yolo_to_corner(box)
    expected_corners = torch.tensor([0.25, 0.25, 0.75, 0.75])
    assert torch.equal(corners, expected_corners)

def test_compute_average_iou(mock_data):
    detect_folder, valid_folder = mock_data
    detections = get_sample_data(detect_folder)
    labels = get_sample_data(valid_folder)
    avg_iou = compute_average_iou(detections, labels)
    expected_iou = (0.64 + 1.0 + 0.64 + 1.0 + 1.0) / 5  # average IoU for all objects
    assert round(avg_iou, 2) == round(expected_iou, 2)

def test_main(mock_data):
    detect_folder, valid_folder = mock_data
    mock_args = Mock()
    mock_args.detect_folder = detect_folder
    mock_args.valid_folder = valid_folder
    with patch('iou2.argparse.ArgumentParser.parse_args', return_value=mock_args):
        with patch('builtins.print') as mock_print:
            main()
            mock_print.assert_called()

def test_get_sample_data_3objs(mock_data):
    detect_folder, valid_folder = mock_data
    detections = get_sample_data(detect_folder)
    labels = get_sample_data(valid_folder)
    detect_3objs = next(tensor for tensor in detections if tensor.shape[0] == 3)
    label_3objs = next(tensor for tensor in labels if tensor.shape[0] == 3)
    assert detect_3objs.shape[0] == 3  # check the number of rows in the tensor
    assert label_3objs.shape[0] == 3  # check the number of rows in the tensor
