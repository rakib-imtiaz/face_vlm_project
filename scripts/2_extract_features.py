#!/usr/bin/env python3
"""
Script to extract facial features from images using various VLM models
"""
import os
import argparse
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM
import csv

# Create directories
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def get_image_paths(dataset_name):
    """Get all image paths from dataset directory"""
    image_paths = []
    
    # Handle different dataset locations
    if dataset_name == 'celebN':
        dataset_dir = os.path.join(PROJECT_DIR, 'celebN')  # celebN is in the root directory
    else:
        dataset_dir = os.path.join(DATA_DIR, dataset_name)  # other datasets are in data directory
    
    if not os.path.exists(dataset_dir):
        print(f"Warning: Dataset directory {dataset_dir} does not exist!")
        return image_paths
        
    print(f"Scanning dataset directory: {dataset_dir}")
    
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('.'):
                image_paths.append(os.path.join(root, file))
                
    print(f"Found {len(image_paths)} images in {dataset_dir}")
    return image_paths

class VLMExtractor:
    """Base class for VLM feature extractors"""
    def __init__(self, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None
        
    def load_model(self):
        """Load model and processor"""
        raise NotImplementedError("Subclasses must implement load_model")
        
    def extract_features(self, image_path):
        """Extract features from an image"""
        raise NotImplementedError("Subclasses must implement extract_features")

class PaliGemmaExtractor(VLMExtractor):
    """Feature extractor using PaliGemma model"""
    def load_model(self):
        print(f"Loading PaliGemma model...")
        self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
        self.model = AutoModelForCausalLM.from_pretrained("google/paligemma-3b-mix-224", 
                                                           torch_dtype=torch.bfloat16,
                                                           device_map="auto")
        print("PaliGemma model loaded")
        
    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            prompt = "Describe this person's facial features in detail:"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            generated_ids = self.model.generate(
                **inputs,
                max_length=100,
                do_sample=False
            )
            
            description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            description = description.replace(prompt, "").strip()
            return description
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return "Error: Could not extract features"

class FlorenceExtractor(VLMExtractor):
    """Feature extractor using Florence model"""
    def load_model(self):
        print(f"Loading Florence model...")
        self.processor = AutoProcessor.from_pretrained("microsoft/florence-2-base")
        self.model = AutoModel.from_pretrained("microsoft/florence-2-base")
        self.model.to(self.device)
        print("Florence model loaded")
        
    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            prompt = "Describe this person's facial features in detail:"
            # Since Florence is more of a vision backbone, we'll get embeddings and use a template
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Florence is primarily for embeddings, so we'll provide a structured description
            # In a real implementation, you might want to use these embeddings with a text model
            return f"Florence extracted visual features for face in {os.path.basename(image_path)}. " \
                   f"These features can be used for face recognition and description generation."
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return "Error: Could not extract features"

class CogVLMExtractor(VLMExtractor):
    """Feature extractor using CogVLM model"""
    def load_model(self):
        print(f"Loading CogVLM model...")
        self.processor = AutoProcessor.from_pretrained("THUDM/cogvlm-chat-hf")
        self.model = AutoModelForCausalLM.from_pretrained("THUDM/cogvlm-chat-hf", 
                                                           torch_dtype=torch.float16,
                                                           device_map="auto")
        print("CogVLM model loaded")
        
    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            prompt = "Describe this person's facial features in detail. Include information about eyes, nose, mouth, face shape, and any distinctive features."
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            generated_ids = self.model.generate(
                **inputs,
                max_length=200,
                do_sample=False
            )
            
            description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Clean up the response to get just the description
            if prompt in description:
                description = description.split(prompt)[1].strip()
            return description
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return "Error: Could not extract features"

class LlamaVisionExtractor(VLMExtractor):
    """Feature extractor using Llama-3.2-Vision model"""
    def load_model(self):
        print(f"Loading Llama-3.2-Vision model...")
        self.processor = AutoProcessor.from_pretrained("meta-llama/Meta-Llama-3.2-8B-Vision")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.2-8B-Vision",
                                                           torch_dtype=torch.bfloat16,
                                                           device_map="auto")
        print("Llama-3.2-Vision model loaded")
        
    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            prompt = "Describe the facial features of the person in this image in detail."
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            generated_ids = self.model.generate(
                **inputs,
                max_length=150,
                do_sample=False
            )
            
            description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Clean up to get just the response
            if prompt in description:
                description = description.split(prompt)[1].strip()
            return description
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return "Error: Could not extract features"

class LLaVAExtractor(VLMExtractor):
    """Feature extractor using LLaVA model"""
    def load_model(self):
        print(f"Loading LLaVA model...")
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.model = AutoModelForCausalLM.from_pretrained("llava-hf/llava-1.5-7b-hf",
                                                           torch_dtype=torch.float16,
                                                           device_map="auto")
        print("LLaVA model loaded")
        
    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            prompt = "USER: <image>\nDescribe the facial features of this person in detail. Include information about eyes, hair, nose, mouth, face shape, and any distinctive features.\nASSISTANT:"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            generated_ids = self.model.generate(
                **inputs,
                max_length=200,
                do_sample=False
            )
            
            description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Clean up to get just the assistant's response
            if "ASSISTANT:" in description:
                description = description.split("ASSISTANT:")[1].strip()
            return description
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return "Error: Could not extract features"

def main():
    parser = argparse.ArgumentParser(description="Extract facial features using VLM models")
    parser.add_argument('--dataset', type=str, default='celebN', choices=['lfw', 'celeba', 'celebN'],
                       help='Dataset directory name (lfw, celeba, or celebN)')
    parser.add_argument('--models', type=str, nargs='+', 
                        choices=['paligemma', 'florence', 'cogvlm', 'llama', 'llava', 'all'],
                        default=['paligemma'],
                        help='VLM models to use')
    parser.add_argument('--limit', type=int, default=10,
                        help='Limit number of images to process')
    args = parser.parse_args()
    
    # Check if 'all' is selected
    if 'all' in args.models:
        models = ['paligemma', 'florence', 'cogvlm', 'llama', 'llava']
    else:
        models = args.models
    
    # Get all image paths from the dataset
    image_paths = get_image_paths(args.dataset)
    
    if not image_paths:
        print(f"No images found in the dataset directory. Please check that the {args.dataset} dataset exists.")
        return
    
    # Limit the number of images to process
    if len(image_paths) > args.limit:
        print(f"Limiting to {args.limit} images")
        image_paths = image_paths[:args.limit]
    
    # Extract features for each model
    for model_name in models:
        print(f"\nProcessing with {model_name} model...")
        
        # Create directory for the model outputs
        model_output_dir = os.path.join(OUTPUTS_DIR, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Create appropriate feature extractor
        if model_name == 'paligemma':
            extractor = PaliGemmaExtractor(model_name)
        elif model_name == 'florence':
            extractor = FlorenceExtractor(model_name)
        elif model_name == 'cogvlm':
            extractor = CogVLMExtractor(model_name)
        elif model_name == 'llama':
            extractor = LlamaVisionExtractor(model_name)
        elif model_name == 'llava':
            extractor = LLaVAExtractor(model_name)
        
        # Load the model
        extractor.load_model()
        
        # Extract features for each image
        descriptions = {}
        csv_data = []
        
        for image_path in tqdm(image_paths, desc=f"Extracting features with {model_name}"):
            try:
                # Get filename (for saving)
                image_name = os.path.basename(image_path)
                
                # Extract features
                description = extractor.extract_features(image_path)
                
                # Store description
                descriptions[image_name] = description
                
                # Append to CSV data
                csv_data.append({
                    'image': image_name,
                    'description': description
                })
                
                print(f"Processed {image_name}: {description[:50]}...")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        # Save descriptions as JSON
        json_path = os.path.join(model_output_dir, 'facial_descriptions.json')
        with open(json_path, 'w') as f:
            json.dump(descriptions, f, indent=4)
            
        # Save descriptions as CSV
        csv_path = os.path.join(model_output_dir, 'facial_descriptions.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'description'])
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"Saved {len(descriptions)} descriptions to {json_path} and {csv_path}")
    
    print("\nFeature extraction complete!")
    print(f"Check the outputs in {OUTPUTS_DIR} directory")

if __name__ == "__main__":
    main() 