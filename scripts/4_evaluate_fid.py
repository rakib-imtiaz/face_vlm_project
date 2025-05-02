#!/usr/bin/env python3
"""
Script to calculate FID scores between original and generated images
"""
import os
import argparse
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

# Try to import pytorch_fid for more accurate FID calculation
try:
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
    HAS_PYTORCH_FID = True
except ImportError:
    print("pytorch_fid not installed, using custom FID implementation")
    HAS_PYTORCH_FID = False

# Create directories
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'outputs')
GENERATED_DIR = os.path.join(OUTPUTS_DIR, 'generated_images')
EVAL_DIR = os.path.join(OUTPUTS_DIR, 'evaluation')
os.makedirs(EVAL_DIR, exist_ok=True)

# Define transforms
preprocess = Compose([
    Resize(299),  # Inception V3 input size
    CenterCrop(299),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_inception_model():
    """Get InceptionV3 model for feature extraction"""
    inception_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
    if torch.cuda.is_available():
        inception_model = inception_model.cuda()
    return inception_model

def get_activations(files, model, batch_size=50, dims=2048):
    """Calculate activations of inception model for images"""
    model.eval()
    
    n_batches = len(files) // batch_size + 1
    n_used_imgs = len(files)
    pred_arr = np.empty((n_used_imgs, dims))
    
    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(files))
        
        if start >= len(files):
            break
            
        batch = []
        for f in files[start:end]:
            try:
                img = Image.open(f).convert('RGB')
                batch.append(preprocess(img))
            except Exception as e:
                print(f"Error processing {f}: {e}")
                n_used_imgs -= 1
                
        if len(batch) == 0:
            continue
            
        batch = torch.stack(batch).cuda() if torch.cuda.is_available() else torch.stack(batch)
        
        with torch.no_grad():
            pred = model(batch)[0]
        
        # If model output is not scalar, apply avg_pool and flatten to vector
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start:start + pred.shape[0]] = pred
    
    return pred_arr[:n_used_imgs]

def calculate_statistics(activations):
    """Calculate mean and covariance statistics"""
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def calculate_fid(real_path, generated_path, model_name=None, generator_name=None):
    """Calculate FID score between original and generated images"""
    if not HAS_PYTORCH_FID:
        print("For accurate FID calculation, install pytorch-fid: pip install pytorch-fid")
    
    # Get all image files
    original_files = []
    generated_files = []
    
    # Get generation info if available
    info_path = os.path.join(generated_path, 'generation_info.csv')
    if os.path.exists(info_path):
        info_df = pd.read_csv(info_path)
        
        # Match original images with their generated counterparts
        for _, row in info_df.iterrows():
            original_image_path = os.path.join(real_path, row['original_image'])
            generated_image_path = os.path.join(generated_path, row['generated_image'])
            
            if os.path.exists(original_image_path) and os.path.exists(generated_image_path):
                original_files.append(original_image_path)
                generated_files.append(generated_image_path)
    else:
        # If no info file, scan directories and try to match by filename
        for root, _, files in os.walk(generated_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    if file.startswith('generated_'):
                        original_name = file.replace('generated_', '')
                        original_path = os.path.join(real_path, original_name)
                        
                        if os.path.exists(original_path):
                            original_files.append(original_path)
                            generated_files.append(os.path.join(root, file))
    
    if len(original_files) == 0 or len(generated_files) == 0:
        print(f"No matching image pairs found in {real_path} and {generated_path}")
        return None
        
    print(f"Found {len(original_files)} matching image pairs")
    
    # Load inception model
    inception_model = get_inception_model()
    
    # Get activations
    print("Calculating activations for original images...")
    real_activations = get_activations(original_files, inception_model)
    
    print("Calculating activations for generated images...")
    generated_activations = get_activations(generated_files, inception_model)
    
    # Calculate statistics
    print("Calculating FID...")
    mu1, sigma1 = calculate_statistics(real_activations)
    mu2, sigma2 = calculate_statistics(generated_activations)
    
    # Calculate FID
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    # Print results
    print(f"FID: {fid_value:.4f}")
    
    # Create a results dictionary
    result = {
        'vlm_model': model_name,
        'generator_model': generator_name,
        'fid_score': float(fid_value),
        'num_images': len(original_files)
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Calculate FID scores between original and generated images")
    parser.add_argument('--dataset', type=str, default='lfw', help='Dataset directory name in data/')
    parser.add_argument('--vlm_models', type=str, nargs='+', 
                        choices=['paligemma', 'florence', 'cogvlm', 'llama', 'llava', 'all'],
                        default=['all'],
                        help='VLM models to evaluate')
    parser.add_argument('--generators', type=str, nargs='+', 
                        choices=['stable-diffusion', 'sdxl-turbo', 'realistic-vision', 'portrait-plus', 'pixart-alpha', 'all'],
                        default=['all'],
                        help='Image generators to evaluate')
    args = parser.parse_args()
    
    # Check if pytorch_fid is installed
    if not HAS_PYTORCH_FID:
        print("Warning: pytorch_fid not installed. For accurate FID scores, install with: pip install pytorch-fid")
    
    # Check if 'all' is selected for VLM models
    if 'all' in args.vlm_models:
        vlm_models = ['paligemma', 'florence', 'cogvlm', 'llama', 'llava']
    else:
        vlm_models = args.vlm_models
    
    # Check if 'all' is selected for generators
    if 'all' in args.generators:
        generators = ['stable-diffusion', 'sdxl-turbo', 'realistic-vision', 'portrait-plus', 'pixart-alpha']
    else:
        generators = args.generators
    
    # Get dataset directory
    dataset_dir = os.path.join(DATA_DIR, args.dataset)
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} does not exist. Please run 1_download_dataset.py first.")
        return
    
    # Calculate FID for each VLM and generator combination
    results = []
    
    for vlm_model in vlm_models:
        for generator in generators:
            generation_dir = os.path.join(GENERATED_DIR, f"{vlm_model}_to_{generator}")
            
            if not os.path.exists(generation_dir):
                print(f"Generated images directory {generation_dir} does not exist.")
                continue
                
            print(f"\nEvaluating {vlm_model} with {generator}...")
            result = calculate_fid(dataset_dir, generation_dir, vlm_model, generator)
            
            if result:
                results.append(result)
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_file = os.path.join(EVAL_DIR, 'fid_scores.csv')
        results_df.to_csv(results_file, index=False)
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Group by VLM model and plot
        for vlm_model in vlm_models:
            model_results = results_df[results_df['vlm_model'] == vlm_model]
            if not model_results.empty:
                plt.bar(
                    [f"{vlm_model}_{gen}" for gen in model_results['generator_model']], 
                    model_results['fid_score'], 
                    label=vlm_model
                )
        
        plt.xlabel('Model Combination')
        plt.ylabel('FID Score (lower is better)')
        plt.title('FID Scores by Model Combination')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(EVAL_DIR, 'fid_scores.png'))
        
        print(f"\nResults saved to {results_file}")
        print(f"Plot saved to {os.path.join(EVAL_DIR, 'fid_scores.png')}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main() 