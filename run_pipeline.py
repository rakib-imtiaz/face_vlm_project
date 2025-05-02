#!/usr/bin/env python3
"""
Main script to run the entire facial feature extraction and generation pipeline
"""
import os
import argparse
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and print its output"""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}\n")
    print(f"Running command: {' '.join(command)}")
    
    start_time = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    end_time = time.time()
    
    print(f"\nCommand completed in {end_time - start_time:.2f} seconds with exit code {process.returncode}")
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description="Run the entire facial feature extraction and generation pipeline")
    parser.add_argument('--dataset', type=str, default='celebN', choices=['lfw', 'celeba', 'celebN'],
                        help='Dataset to use (lfw, celeba, or celebN)')
    parser.add_argument('--vlm_models', type=str, nargs='+', 
                        choices=['paligemma', 'florence', 'cogvlm', 'llama', 'llava', 'all'],
                        default=['all'],
                        help='VLM models to use for feature extraction')
    parser.add_argument('--generators', type=str, nargs='+', 
                        choices=['stable-diffusion', 'sdxl-turbo', 'realistic-vision', 'portrait-plus', 'pixart-alpha', 'all'],
                        default=['all'],
                        help='Image generators to use')
    parser.add_argument('--extract_limit', type=int, default=100,
                        help='Limit number of images to process for feature extraction')
    parser.add_argument('--generate_limit', type=int, default=10,
                        help='Limit number of descriptions to process for image generation')
    parser.add_argument('--skip_download', action='store_true', default=True,
                        help='Skip the dataset download step')
    parser.add_argument('--skip_extraction', action='store_true',
                        help='Skip the feature extraction step')
    parser.add_argument('--skip_generation', action='store_true',
                        help='Skip the image generation step')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip the FID evaluation step')
    args = parser.parse_args()
    
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Build common args for VLM models and generators
    vlm_models_arg = ' '.join(args.vlm_models)
    generators_arg = ' '.join(args.generators)
    
    # Create necessary directories
    os.makedirs(os.path.join(script_dir, 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(script_dir, 'outputs', 'evaluation'), exist_ok=True)
    os.makedirs(os.path.join(script_dir, 'outputs', 'generated_images'), exist_ok=True)
    
    # STEP 2: Extract facial features (skip STEP 1: Download dataset)
    if not args.skip_extraction:
        extract_cmd = [
            "python", str(script_dir / "scripts" / "2_extract_features.py"),
            "--dataset", args.dataset,
            "--models"] + args.vlm_models + [
            "--limit", str(args.extract_limit)
        ]
        status = run_command(extract_cmd, "Extract Facial Features using VLMs")
        if status != 0:
            print("Feature extraction failed. Exiting.")
            return
    
    # STEP 3: Generate images
    if not args.skip_generation:
        generate_cmd = [
            "python", str(script_dir / "scripts" / "3_generate_images.py"),
            "--dataset", args.dataset,
            "--vlm_model"] + (["all"] if "all" in args.vlm_models else args.vlm_models) + [
            "--generators"] + args.generators + [
            "--limit", str(args.generate_limit)
        ]
        status = run_command(generate_cmd, "Generate Images from Facial Descriptions")
        if status != 0:
            print("Image generation failed. Exiting.")
            return
    
    # STEP 4: Evaluate FID scores
    if not args.skip_evaluation:
        evaluate_cmd = [
            "python", str(script_dir / "scripts" / "4_evaluate_fid.py"),
            "--dataset", args.dataset,
            "--vlm_models"] + (["all"] if "all" in args.vlm_models else args.vlm_models) + [
            "--generators"] + args.generators
        status = run_command(evaluate_cmd, "Evaluate Generated Images using FID Score")
        if status != 0:
            print("FID evaluation failed.")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80 + "\n")
    
    print("Pipeline summary:")
    if not args.skip_extraction:
        print(f"✓ Facial features extracted from {args.extract_limit} images using {vlm_models_arg}")
    if not args.skip_generation:
        print(f"✓ Images generated from {args.generate_limit} descriptions using {generators_arg}")
    if not args.skip_evaluation:
        print(f"✓ FID scores calculated for generated images")
    
    print("\nOutput files can be found in the 'outputs' directory")
    
    # Show visualization instructions
    print("\nTo visualize the results:")
    print("1. Check the CSV files in 'outputs/<vlm_model>' for extracted features")
    print("2. View generated images in 'outputs/generated_images/<vlm_model>_to_<generator>'")
    print("3. FID scores are available in 'outputs/evaluation/fid_scores.csv'")
    print("4. A bar chart of FID scores is saved in 'outputs/evaluation/fid_scores.png'")

if __name__ == "__main__":
    main() 