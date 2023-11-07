import pytest
import subprocess
from pathlib import Path

# Fixtures for the common paths used in the tests
@pytest.fixture
def yolov5_dir():
    return Path(__file__).parent / 'yolov5'

@pytest.fixture
def data_yaml():
    return Path(__file__).parent / 'dataset.yaml'

@pytest.fixture
def weights():
    return Path(__file__).parent / 'yolov5s-model.pt'

@pytest.fixture
def results_dir(yolov5_dir):
    return yolov5_dir / 'runs/val/conf_matrix_test'

# Test different batch sizes and image sizes
@pytest.mark.parametrize("batch_size", [1, 8, 16])
@pytest.mark.parametrize("img_size", [320, 640, 1280])
def test_various_batch_img_sizes(yolov5_dir, data_yaml, weights, results_dir, batch_size, img_size):
    conf_thres = 0.25
    iou_thres = 0.45
    task = 'val'

    cmd = [
        'python3', 'val.py',
        '--weights', str(weights),
        '--data', str(data_yaml),
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--conf', str(conf_thres),
        '--iou', str(iou_thres),
        '--task', task,
        '--name', 'conf_matrix_test'
    ]

    result = subprocess.run(cmd, cwd=yolov5_dir, stdout=subprocess.PIPE, text=True)
    assert result.returncode == 0, f"val.py failed with batch_size={batch_size}, img_size={img_size}."

    confusion_matrix_file = results_dir / 'confusion_matrix.png'
    assert confusion_matrix_file.is_file(), "Confusion matrix file was not created."

# Test different confidence and IoU thresholds
@pytest.mark.parametrize("conf_thres,iou_thres", [(0.1, 0.4), (0.25, 0.45), (0.5, 0.6)])
def test_various_thresholds(yolov5_dir, data_yaml, weights, results_dir, conf_thres, iou_thres):
    batch_size = 16
    img_size = 640
    task = 'val'

    cmd = [
        'python3', 'val.py',
        '--weights', str(weights),
        '--data', str(data_yaml),
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--conf', str(conf_thres),
        '--iou', str(iou_thres),
        '--task', task,
        '--name', 'conf_matrix_test'
    ]

    result = subprocess.run(cmd, cwd=yolov5_dir, stdout=subprocess.PIPE, text=True)
    assert result.returncode == 0, f"val.py failed with conf_thres={conf_thres}, iou_thres={iou_thres}."

# Test different tasks
@pytest.mark.parametrize("task", ['val', 'test', 'train'])
def test_different_tasks(yolov5_dir, data_yaml, weights, results_dir, task):
    batch_size = 16
    img_size = 640
    conf_thres = 0.25
    iou_thres = 0.45

    cmd = [
        'python3', 'val.py',
        '--weights', str(weights),
        '--data', str(data_yaml),
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--conf', str(conf_thres),
        '--iou', str(iou_thres),
        '--task', task,
        '--name', f'conf_matrix_{task}'
    ]

    result = subprocess.run(cmd, cwd=yolov5_dir, stdout=subprocess.PIPE, text=True)
    assert result.returncode == 0, f"val.py failed with task={task}."

# Test with invalid weight path
def test_invalid_weight_path(yolov5_dir, data_yaml):
    invalid_weights = 'invalid_path.pt'
    batch_size = 16
    img_size = 640
    conf_thres = 0.25
    iou_thres = 0.45
    task = 'val'

    cmd = [
        'python3', 'val.py',
        '--weights', invalid_weights,
        '--data', str(data_yaml),
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--conf', str(conf_thres),
        '--iou', str(iou_thres),
        '--task', task,
        '--name', 'conf_matrix_invalid_weights'
    ]

    result = subprocess.run(cmd, cwd=yolov5_dir, stdout=subprocess.PIPE, text=True)
    assert result.returncode != 0, "The script should fail with an invalid weight path."