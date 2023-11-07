# test_conf_matrix.py

import pytest
from conf_matrix import generate_confusion_matrix
from pathlib import Path

@pytest.fixture
def yolov5_dir():
    return Path(__file__).parent / 'yolov5'

@pytest.fixture
def data_yaml():
    return Path(__file__).parent / 'dataset.yaml'

@pytest.fixture
def weights():
    return Path(__file__).parent / 'yolov5s-model.pt'

def test_generate_confusion_matrix(yolov5_dir, data_yaml, weights):
    batch_size = 16
    img_size = 640
    conf_thres = 0.25
    iou_thres = 0.45
    task = 'val'

    # Call the function with the fixtures
    generate_confusion_matrix(yolov5_dir, data_yaml, weights, batch_size, img_size, conf_thres, iou_thres, task)

    results_dir = yolov5_dir / 'runs/val/conf_matrix'
    confusion_matrix_file = results_dir / 'confusion_matrix.png'
    assert confusion_matrix_file.is_file(), "Confusion matrix file was not created."
