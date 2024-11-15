#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
from loguru import logger

def convert_to_16bit(input_file, output_file):
    command = [
        "ffmpeg",
        "-i", str(input_file),
        "-acodec", "pcm_s16le",  # 轉換為 16-bit PCM
        str(output_file)
    ]
    subprocess.run(command, check=True)

def convert_noise_files_to_16bit(noise_files, output_dir):
    logger.info("Starting conversion of noise files to 16-bit PCM...")
    converted_files = []
    os.makedirs(output_dir, exist_ok=True)
    
    for noise_file in noise_files:
        # 保持原始檔案名稱不變，將其存儲在 audio_16 資料夾中
        output_file = output_dir / noise_file.name
        logger.debug(f"Converting {noise_file} to {output_file}")
        convert_to_16bit(noise_file, output_file)
        converted_files.append(output_file)
    
    logger.info(f"Completed conversion of {len(noise_files)} noise files to 16-bit PCM.")
    return converted_files

def read_file_list(file_list_path):
    with open(file_list_path, "r") as f:
        files = [Path(line.strip()) for line in f]
    return files

def main():
    DATA_DIR = Path("data").resolve()
    WHAM_DIR = DATA_DIR / "high_res_wham/audio"
    noise_files_list = read_file_list(DATA_DIR / "noise_files.txt")
    output_dir = DATA_DIR / "high_res_wham" / "audio_16"
    convert_noise_files_to_16bit(noise_files_list, output_dir)

if __name__ == "__main__":
    main()