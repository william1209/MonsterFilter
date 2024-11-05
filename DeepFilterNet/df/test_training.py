import numpy as np
import os
import sys
from pathlib import Path
import h5py
import json
from loguru import logger


def create_test_data():
    # 創建目錄
    os.makedirs("test_data", exist_ok=True)
    os.makedirs("test_models", exist_ok=True)
    
    # 生成 dataset.cfg
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
    
    # 寫入 dataset.cfg
    with open("test_data/dataset.cfg", "w") as f:
        json.dump(dataset_config, f, indent=4)
    logger.info("Created dataset.cfg")
    
    # 生成訓練配置
    train_config = """[train]
seed = 42
device = 
model = deepfilternet
mask_only = False
df_only = False
jit = False
batch_size = 64
batch_size_eval = 0
overfit = False
num_workers = 8
max_sample_len_s = 5.0
num_prefetch_batches = 32
global_ds_sampling_f = 1.0
dataloader_snrs = -5,0,5,10,20,40
dataloader_gains = -6,0,6
batch_size_scheduling = 
max_epochs = 10
validation_criteria = loss
validation_criteria_rule = min
early_stopping_patience = 5
start_eval = False

[df]
sr = 48000
fft_size = 960
hop_size = 480
nb_erb = 32
nb_df = 96
norm_tau = 1
lsnr_max = 35
lsnr_min = -15
min_nb_erb_freqs = 2
df_order = 5
df_lookahead = 0
pad_mode = input

[deepfilternet]
conv_lookahead = 0
conv_ch = 16
conv_depthwise = True
convt_depthwise = True
conv_kernel = 1,3
convt_kernel = 1,3
conv_kernel_inp = 3,3
emb_hidden_dim = 256
emb_num_layers = 2
emb_gru_skip_enc = none
emb_gru_skip = none
df_hidden_dim = 256
df_gru_skip = none
df_pathway_kernel_size_t = 1
enc_concat = False
df_num_layers = 3
df_n_iter = 1
linear_groups = 1
enc_linear_groups = 16
mask_pf = False
pf_beta = 0.02
lsnr_dropout = False
conv_k_enc = 2
conv_k_dec = 1
conv_width_factor = 1
conv_dec_mode = transposed
gru_groups = 1
group_shuffle = True
dfop_method = real_unfold

# SKNet 配置
use_sknet = true
sk_kernels = 3,5
sk_reduction = 16
sk_groups = 4
sk_min_width = 32
log_model_summary = true

[distortion]
p_reverb = 0.2
p_bandwidth_ext = 0.0
p_clipping = 0.0
p_zeroing = 0.0
p_air_absorption = 0.0
p_interfer_sp = 0.0

[optim]
lr = 0.0005
momentum = 0
weight_decay = 0.05
optimizer = adamw
lr_min = 1e-06
lr_warmup = 0.0001
warmup_epochs = 3
lr_cycle_mul = 1.0
lr_cycle_decay = 0.5
lr_cycle_epochs = -1
weight_decay_end = -1

[maskloss]
factor = 0
mask = iam
gamma = 0.6
gamma_pred = 0.6
f_under = 2
max_freq = 0

[spectralloss]
factor_magnitude = 0
factor_complex = 0
factor_under = 1
gamma = 1

[multiresspecloss]
factor = 0
factor_complex = 0
gamma = 1
fft_sizes = 512,1024,2048

[sdrloss]
factor = 0

[localsnrloss]
factor = 0.0005

[asrloss]
factor = 0
factor_lm = 0
loss_lm = CrossEntropy
model = base.en
"""

    # 寫入配置文件
    config_path = os.path.join("test_models", "config.ini")
    with open(config_path, "w") as f:
        f.write(train_config)
    logger.info(f"Created config file at {config_path}")
    logger.info("sdfsdfsdfsdfs")
    
    # 創建測試數據
    n_samples = 48000  # 1秒的音頻
    if not os.path.exists("test_data/test_speech.hdf5"):
        with h5py.File("test_data/test_speech.hdf5", "w") as f:
            speech_data = np.random.randn(n_samples) * 0.1
            speech_group = f.create_group("speech")
            speech_group.create_dataset("files", data=[speech_data.astype(np.float32)])
            speech_group.attrs["working_dir"] = "test_data"
            speech_group.attrs["sr"] = 48000
            speech_group.attrs["max_freq"] = 24000
            speech_group.attrs["codec"] = "pcm"
            speech_group.attrs["dtype"] = "float32"
            
    if not os.path.exists("test_data/test_noise.hdf5"):
        with h5py.File("test_data/test_noise.hdf5", "w") as f:
            noise_data = np.random.randn(n_samples) * 0.05
            noise_group = f.create_group("noise")
            noise_group.create_dataset("files", data=[noise_data.astype(np.float32)])
            noise_group.attrs["working_dir"] = "test_data"
            noise_group.attrs["sr"] = 48000
            noise_group.attrs["max_freq"] = 24000
            noise_group.attrs["codec"] = "pcm"
            noise_group.attrs["dtype"] = "float32"
            
    logger.info("Created test datasets")

if __name__ == "__main__":
    # 強制重新加載模塊
    from config import config
    import importlib
    import train
    import model
    import deepfilternet
    from argparse import Namespace
    
    importlib.reload(train)
    importlib.reload(model)
    importlib.reload(deepfilternet)
    
    # 初始化日誌
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    logger.add("test_models/train.log", level="DEBUG")
    
    # 添加版本檢查
    logger.info(f"Current df.train version: {train.__file__}")
    logger.info(f"Current df.model version: {model.__file__}")

    # 檢查 Rust 組件
    import libdf
    import libdfdata
    logger.info(f"libdf path: {libdf.__file__}")
    logger.info(f"libdfdata path: {libdfdata.__file__}")
    
    # 創建測試數據
    create_test_data()

    
    config_path = os.path.join("test_models", "config.ini")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    # 在導入其他模塊之前加載配置
    config.load(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    

    # 運行訓練
    from train import main
    args = Namespace(
        data_config_file="test_data/dataset.cfg",  # 改為 data_config_file
        data_dir="test_data",
        base_dir="test_models",  # 改為 base_dir
        host_batchsize_config=None,
        debug=True,
        resume=False,  # 使用 resume 而不是 no_resume
        log_level=None  # 添加 log_level 參數
    )
    
    logger.info("Starting training with fresh modules")
    main(args)
