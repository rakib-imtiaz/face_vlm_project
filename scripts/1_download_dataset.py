#!/usr/bin/env python3
"""
Script to download a medium-sized face recognition dataset
"""
import os
import argparse
import requests
import zipfile
from tqdm import tqdm
import shutil

# Create directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, output_path):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path) as pbar:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)

def download_lfw_dataset():
    """Download LFW dataset - a medium-sized face recognition dataset"""
    print("Downloading LFW dataset...")
    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    lfw_path = os.path.join(DATA_DIR, "lfw.tgz")
    lfw_dir = os.path.join(DATA_DIR, "lfw")
    
    # Download the dataset
    if not os.path.exists(lfw_path):
        download_file(lfw_url, lfw_path)
    
    # Extract the dataset
    if not os.path.exists(lfw_dir):
        print("Extracting LFW dataset...")
        import tarfile
        with tarfile.open(lfw_path) as tar:
            tar.extractall(path=DATA_DIR)
    
    print(f"LFW dataset downloaded and extracted to {lfw_dir}")
    return lfw_dir

def download_celeba_dataset():
    """Download a subset of CelebA dataset (alternative)"""
    print("Downloading subset of CelebA dataset...")
    # Note: This is a simplified approach; in practice, you might want to use an official API
    celeba_sample_url = "https://mmlab.ie.cuhk.edu.hk/projects/CelebA/files/img_align_celeba.zip"
    
    # For this project we'll just use a very small subset (first 1000 images)
    celeba_path = os.path.join(DATA_DIR, "celeba.zip")
    celeba_dir = os.path.join(DATA_DIR, "celeba")
    extracted_dir = os.path.join(DATA_DIR, "img_align_celeba")
    
    if not os.path.exists(celeba_dir):
        os.makedirs(celeba_dir, exist_ok=True)
    
    # Note: CelebA is quite large, so in practice you would download it separately
    # or use a managed dataset from Hugging Face or similar
    
    print(f"For CelebA, please manually download a subset from: {celeba_sample_url}")
    print(f"and place a manageable number of images in: {celeba_dir}")
    
    return celeba_dir

def main():
    parser = argparse.ArgumentParser(description="Download face dataset")
    parser.add_argument('--dataset', type=str, choices=['lfw', 'celeba'], default='lfw',
                        help='Dataset to download (lfw or celeba)')
    args = parser.parse_args()
    
    if args.dataset == 'lfw':
        dataset_dir = download_lfw_dataset()
    else:
        dataset_dir = download_celeba_dataset()
    
    print(f"Dataset downloaded to: {dataset_dir}")
    print("Number of images:", sum(len(files) for _, _, files in os.walk(dataset_dir)))

if __name__ == "__main__":
    main() 