#!/usr/bin/env python3
import os
import json
import subprocess
from pathlib import Path
from loguru import logger
import sys
import argparse

# 設定基本路徑
MODEL_DIR = Path("models").resolve()
DATA_DIR = Path("data").resolve()
VCTK_DIR = DATA_DIR / "VCTK-Corpus-0.92/wav48_silence_trimmed"
WHAM_DIR = DATA_DIR / "high_res_wham/audio_16"
HDF5_DIR = DATA_DIR / "hdf5"
os.makedirs(HDF5_DIR, exist_ok=True)

def read_file_list(file_list_path):
    with open(file_list_path, "r") as f:
        files = [DATA_DIR / line.strip() for line in f]
    return files

# 1. 建立檔案清單
def create_file_lists():
    logger.info("Creating file lists...")
    speech_files = DATA_DIR / "speech_files.txt"
    noise_files = DATA_DIR / "noise_files.txt"
    speech_count = 0
    noise_count = 0

     # 收集 VCTK 語音檔案 - 使用相對路徑
    all_speech_files = []
    for speaker_dir in VCTK_DIR.glob("p*"):
        for audio_file in speaker_dir.glob("*.flac"):
            rel_path = audio_file.relative_to(DATA_DIR)
            all_speech_files.append(rel_path)
    
    # 只使用前四分之一的語音檔案
    quarter_speech_files = all_speech_files[:len(all_speech_files) // 4]
    with open(speech_files, "w") as f:
        for rel_path in quarter_speech_files:
            f.write(f"{rel_path}\n")
            speech_count += 1
    logger.info(f"Collected {speech_count} VCTK speech files from {VCTK_DIR}")
    
    # 收集 WHAM 噪音檔案 - 使用相對路徑
    all_noise_files = []
    for audio_file in WHAM_DIR.glob("*.wav"):
        rel_path = audio_file.relative_to(DATA_DIR)
        all_noise_files.append(rel_path)
    
    # 只使用前四分之一的噪音檔案
    quarter_noise_files = all_noise_files[:len(all_noise_files) // 4]
    with open(noise_files, "w") as f:
        for rel_path in quarter_noise_files:
            f.write(f"{rel_path}\n")
            noise_count += 1
    logger.info(f"Collected {noise_count} WHAM noise files from {WHAM_DIR}")
    
    
    # 收集 VCTK 語音檔案 - 使用相對路徑
    '''
    with open(speech_files, "w") as f:
        for speaker_dir in VCTK_DIR.glob("p*"):
            for audio_file in speaker_dir.glob("*.flac"):
                rel_path = audio_file.relative_to(DATA_DIR)
                f.write(f"{rel_path}\n")
                speech_count += 1
    logger.info(f"Collected {speech_count} VCTK speech files from {VCTK_DIR}")
    '''
    '''
    # 收集 WHAM 噪音檔案 - 使用相對路徑
    with open(noise_files, "w") as f:
        for audio_file in WHAM_DIR.glob("*.wav"):
            rel_path = audio_file.relative_to(DATA_DIR)
            f.write(f"{rel_path}\n")
            noise_count += 1
    logger.info(f"Collected {noise_count} WHAM noise files from {WHAM_DIR}")
    '''
    logger.info(f"Total files: {speech_count + noise_count} (Speech: {speech_count}, Noise: {noise_count})")
    return speech_files, noise_files, None
    
# 2. 準備 HDF5 資料集
def prepare_datasets(speech_files, noise_files, convert_to_16bit):
    logger.info("Preparing HDF5 datasets...")
    
    from prepare_data import main as prepare_data_main
    import sys
    
    # 讀取檔案清單
    speech_files_list = read_file_list(speech_files)
    noise_files_list = read_file_list(noise_files)
    
    # 根據參數決定是否進行轉換
    if convert_to_16bit:
        logger.info("Converting WHAM noise files to 16-bit PCM format using external script...")
        subprocess.run(["python", "convert_to_16bit.py"], check=True)
        # 更新 noise_files_list 以指向轉換後的檔案
        noise_files_list = [DATA_DIR / "audio_16" / file.name for file in noise_files_list]
    
    # 將檔案清單寫入臨時檔案
    temp_noise_files = DATA_DIR / "temp_noise_files.txt"
    with open(temp_noise_files, "w") as f:
        for file in noise_files_list:
            f.write(f"{file.relative_to(DATA_DIR)}\n")
    
    # 處理 VCTK 語音資料
    logger.info("Processing speech data...")
    sys.argv = [
        "prepare_data.py",
        "--sr", "48000",
        "--dtype", "float32",
        "--num_workers", "4",
        "--mono",
        "speech",
        str(speech_files),
        str(HDF5_DIR / "speech.hdf5")
    ]
    prepare_data_main()
    
    # 處理 WHAM 噪音資料
    logger.info("Processing noise data...")
    sys.argv = [
        "prepare_data.py",
        "--sr", "48000",
        "--dtype", "float32",
        "--num_workers", "4",
        "noise", 
        str(temp_noise_files),
        str(HDF5_DIR / "noise.hdf5")
    ]
    prepare_data_main()

# 3. 建立資料集配置檔
def create_dataset_config():
    logger.info("Creating dataset config...")
    config = {
        "train": [
            ["speech.hdf5", 1.0],
            ["noise.hdf5", 1.0],
        ],
        "valid": [
            ["speech.hdf5", 1.0],
            ["noise.hdf5", 1.0],
        ],
        "test": [
            ["speech.hdf5", 1.0],
            ["noise.hdf5", 1.0],
        ]
    }
    
    config_path = DATA_DIR / "dataset.cfg"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Created dataset config at {config_path}")
    
    # 寫入訓練配置
    train_config = """[train]
seed = 42
device = 
model = deepfilternet
mask_only = False
df_only = False
jit = False
batch_size = 128
batch_size_eval = 0
overfit = False
num_workers = 12
max_sample_len_s = 5.0
num_prefetch_batches = 64
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
p_reverb = 0.0
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
    
    train_config_path = MODEL_DIR / "config.ini"
    with open(train_config_path, "w") as f:
        f.write(train_config)
    logger.info(f"Created train config at {train_config_path}")
    
    return config_path, train_config_path

# 4. 開始訓練
def start_training(dataset_config):
    logger.info("Starting training...")
    subprocess.run([
        "python", "train.py",
        str(dataset_config),
        str(HDF5_DIR),
        str(MODEL_DIR)
    ])

def main():
    parser = argparse.ArgumentParser(description="DeepFilterNet Training Pipeline")
    parser.add_argument("--convert_to_16bit", action="store_true", help="Convert 24-bit noise files to 16-bit")
    args = parser.parse_args()

    logger.info("Starting DeepFilterNet training pipeline...")
    
    # 檢查結構目錄
    if not VCTK_DIR.exists() or not WHAM_DIR.exists():
        raise ValueError(f"Missing required directories: {VCTK_DIR} or {WHAM_DIR}")
    
    # 執行訓練流程
    #speech_files, noise_files, _ = create_file_lists()
    #prepare_datasets(speech_files, noise_files, args.convert_to_16bit)
    dataset_config, train_config_path = create_dataset_config()
    start_training(dataset_config)

if __name__ == "__main__":
    main()   