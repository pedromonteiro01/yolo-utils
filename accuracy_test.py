# Author: Pedro Monteiro
# Date: November 2023
# Computer Science Engineering MSc
# Aveiro University

import subprocess
from unittest.mock import patch, mock_open
from accuracy import evaluate_model, extract_metrics, extract_details_from_opt, find_best_config

def test_evaluate_model():
    # mock the subprocess.run to return a predetermined output
    mocked_output = "Mocked stderr that would come from val.py"
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stderr=mocked_output)
        result = evaluate_model("fake_weights.pt")
        assert result == mocked_output

def test_extract_metrics():
    # example output from val.py script
    mocked_output = """
    Class     Images  Instances          P          R      mAP50   mAP50-95: 
    all         14         15       0.82      0.528       0.63       0.34
    """
    mAP50 = extract_metrics(mocked_output)
    assert mAP50 == 0.63  # based on the mocked output

def test_extract_details_from_opt():
    # mock the content of opt.yaml
    mocked_yaml_content = """
    epochs: 50
    batch_size: 16
    """
    with patch("builtins.open", mock_open(read_data=mocked_yaml_content)):
        epochs, batch_size = extract_details_from_opt("fake_path")
        assert epochs == 50
        assert batch_size == 16