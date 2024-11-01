import torch
import numpy as np
import os
from pathlib import Path
import h5py
import json

# 創建測試數據目錄
def create_test_data():
    # 創建目錄
    os.makedirs("test_data", exist_ok=True)
    os.makedirs("test_models", exist_ok=True)
    
    # 生成簡單的合成音頻數據
    duration = 1  # 1秒
    sr = 48000
    n_samples = duration * sr
    
    # 創建語音 HDF5
    with h5py.File("test_data/test_speech.hdf5", "w") as f:
        speech_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, n_samples))  # 440Hz sine wave
        speech_group = f.create_group("speech")
        speech_group.create_dataset("files", data=[speech_data.astype(np.float32)])
        speech_group.attrs["working_dir"] = "test_data"
    
    # 創建噪音 HDF5
    with h5py.File("test_data/test_noise.hdf5", "w") as f:
        noise_data = np.random.randn(n_samples) * 0.1  # 白噪音
        noise_group = f.create_group("noise")
        noise_group.create_dataset("files", data=[noise_data.astype(np.float32)])
        noise_group.attrs["working_dir"] = "test_data"

    # 創建數據集配置文件
    dataset_config = {
        "train": [
            ["test_speech.hdf5", 1.0],
            ["test_noise.hdf5", 1.0]
        ],
        "valid": [
            ["test_speech.hdf5", 1.0],
            ["test_noise.hdf5", 1.0]
        ],
        "test": [
            ["test_speech.hdf5", 1.0],
            ["test_noise.hdf5", 1.0]
        ]
    }
    
    with open("test_data/dataset.cfg", "w") as f:
        json.dump(dataset_config, f, indent=4)

    # 創建訓練配置文件
    train_config = """
[train]
epochs = 2
batch_size = 2
num_workers = 1
max_sample_len_s = 1.0
    """
    with open("test_models/train.cfg", "w") as f:
        f.write(train_config)

if __name__ == "__main__":
    # 創建測試數據
    create_test_data()
    
    # 運行訓練
    from df.train import main
    import sys
    sys.argv = [
        "train.py",
        "test_data/dataset.cfg",
        "test_data",
        "test_models",
        "--debug"
    ]
    main()
