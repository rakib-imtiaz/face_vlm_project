#!/usr/bin/env python3
"""
Script to generate images from facial descriptions using text-to-image models
"""
import os
import argparse
import json
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import pandas as pd

# Create directories
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'outputs')
GENERATED_DIR = os.path.join(OUTPUTS_DIR, 'generated_images')
os.makedirs(GENERATED_DIR, exist_ok=True)

def load_descriptions(model_name):
    """Load facial descriptions extracted by a VLM model"""
    description_path = os.path.join(OUTPUTS_DIR, model_name, 'facial_descriptions.json')
    if not os.path.exists(description_path):
        print(f"Descriptions file for {model_name} does not exist. Please run 2_extract_features.py first.")
        return None
    
    with open(description_path, 'r') as f:
        descriptions = json.load(f)
    
    return descriptions

class ImageGenerator:
    """Base class for text-to-image generators"""
    def __init__(self, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        
    def load_model(self):
        """Load model"""
        raise NotImplementedError("Subclasses must implement load_model")
        
    def generate_image(self, description, output_path):
        """Generate image from description"""
        raise NotImplementedError("Subclasses must implement generate_image")

class StableDiffusionGenerator(ImageGenerator):
    """Image generator using Stable Diffusion models"""
    def load_model(self):
        print(f"Loading Stable Diffusion model...")
        # Using Stable Diffusion XL for better face generation
        self.pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.pipeline.to(self.device)
        print("Stable Diffusion model loaded")
        
    def generate_image(self, description, output_path):
        try:
            # Add prefixes and suffixes to improve generation quality
            prompt = f"Portrait photo of a person with {description}, high resolution, detailed facial features, professional photography"
            
            # Generate image
            image = self.pipeline(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
            
            # Save image
            image.save(output_path)
            return True
        except Exception as e:
            print(f"Error generating image: {e}")
            return False

class SDXLTurboGenerator(ImageGenerator):
    """Image generator using SDXL Turbo model"""
    def load_model(self):
        print(f"Loading SDXL Turbo model...")
        # Using SDXL Turbo for faster generation
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.pipeline.to(self.device)
        print("SDXL Turbo model loaded")
        
    def generate_image(self, description, output_path):
        try:
            # Add prefixes and suffixes to improve generation quality
            prompt = f"Portrait photo of a person with {description}, high resolution, detailed facial features, professional photography"
            
            # Generate image with fewer steps due to turbo model
            image = self.pipeline(prompt, num_inference_steps=4, guidance_scale=0).images[0]
            
            # Save image
            image.save(output_path)
            return True
        except Exception as e:
            print(f"Error generating image: {e}")
            return False

class RealisticVisionGenerator(ImageGenerator):
    """Image generator using Realistic Vision model (specialized for faces)"""
    def load_model(self):
        print(f"Loading Realistic Vision model...")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None
        )
        self.pipeline.to(self.device)
        print("Realistic Vision model loaded")
        
    def generate_image(self, description, output_path):
        try:
            # More specific prompt for realistic face generation
            prompt = f"RAW photo, portrait of a person with {description}, 8k uhd, high quality, realistic, photorealistic"
            
            # Generate image
            image = self.pipeline(
                prompt, 
                negative_prompt="deformed, ugly, disfigured, low quality", 
                num_inference_steps=25, 
                guidance_scale=7
            ).images[0]
            
            # Save image
            image.save(output_path)
            return True
        except Exception as e:
            print(f"Error generating image: {e}")
            return False

class PortraitPlusGenerator(ImageGenerator):
    """Image generator using Portrait Plus model"""
    def load_model(self):
        print(f"Loading Portrait Plus model...")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "wavymulder/portraitplus",
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipeline.to(self.device)
        print("Portrait Plus model loaded")
        
    def generate_image(self, description, output_path):
        try:
            # Prompt specifically formatted for Portrait Plus
            prompt = f"portrait photo of a person with {description}, detailed face, beautiful portrait, perfect face, looking at camera"
            
            # Generate image
            image = self.pipeline(
                prompt,
                negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
                num_inference_steps=25,
                guidance_scale=7.5
            ).images[0]
            
            # Save image
            image.save(output_path)
            return True
        except Exception as e:
            print(f"Error generating image: {e}")
            return False

class PixArtAlphaGenerator(ImageGenerator):
    """Image generator using PixArt-Alpha model"""
    def load_model(self):
        print(f"Loading PixArt-Alpha model...")
        self.pipeline = DiffusionPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.pipeline.to(self.device)
        print("PixArt-Alpha model loaded")
        
    def generate_image(self, description, output_path):
        try:
            # Prompt for PixArt-Alpha
            prompt = f"high-quality portrait photograph of a person with {description}, detailed facial features, professional lighting, 8k, realistic"
            
            # Generate image
            image = self.pipeline(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=4.5
            ).images[0]
            
            # Save image
            image.save(output_path)
            return True
        except Exception as e:
            print(f"Error generating image: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Generate images from facial descriptions")
    parser.add_argument('--vlm_model', type=str, choices=['paligemma', 'florence', 'cogvlm', 'llama', 'llava', 'all'],
                        default='paligemma',
                        help='VLM model whose descriptions to use for generation')
    parser.add_argument('--generators', type=str, nargs='+', 
                        choices=['stable-diffusion', 'sdxl-turbo', 'realistic-vision', 'portrait-plus', 'pixart-alpha', 'all'],
                        default=['stable-diffusion'],
                        help='Image generation models to use')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of descriptions to process')
    parser.add_argument('--min_desc_length', type=int, default=20, 
                        help='Minimum description length to consider for generation')
    args = parser.parse_args()
    
    # Check if 'all' is selected for VLM models
    if args.vlm_model == 'all':
        vlm_models = ['paligemma', 'florence', 'cogvlm', 'llama', 'llava']
    else:
        vlm_models = [args.vlm_model]
    
    # Check if 'all' is selected for generators
    if 'all' in args.generators:
        generators = ['stable-diffusion', 'sdxl-turbo', 'realistic-vision', 'portrait-plus', 'pixart-alpha']
    else:
        generators = args.generators
    
    # Generate images for each VLM model and generator combination
    for vlm_model in vlm_models:
        # Load descriptions
        descriptions = load_descriptions(vlm_model)
        if descriptions is None:
            continue
        
        # Filter descriptions
        filtered_descriptions = {}
        for image_name, desc in descriptions.items():
            if len(desc) >= args.min_desc_length and "Error" not in desc:
                filtered_descriptions[image_name] = desc
        
        if len(filtered_descriptions) == 0:
            print(f"No valid descriptions found for {vlm_model}")
            continue
        
        # Limit the number of descriptions
        if args.limit > 0 and args.limit < len(filtered_descriptions):
            image_names = list(filtered_descriptions.keys())[:args.limit]
            limited_descriptions = {name: filtered_descriptions[name] for name in image_names}
        else:
            limited_descriptions = filtered_descriptions
        
        print(f"Using {len(limited_descriptions)} descriptions from {vlm_model}")
        
        # Create generators
        image_generators = {}
        for generator_name in generators:
            if generator_name == 'stable-diffusion':
                image_generators[generator_name] = StableDiffusionGenerator(generator_name)
            elif generator_name == 'sdxl-turbo':
                image_generators[generator_name] = SDXLTurboGenerator(generator_name)
            elif generator_name == 'realistic-vision':
                image_generators[generator_name] = RealisticVisionGenerator(generator_name)
            elif generator_name == 'portrait-plus':
                image_generators[generator_name] = PortraitPlusGenerator(generator_name)
            elif generator_name == 'pixart-alpha':
                image_generators[generator_name] = PixArtAlphaGenerator(generator_name)
        
        # Load generators
        for generator_name, generator in image_generators.items():
            generator.load_model()
            
            # Create output directory
            output_dir = os.path.join(GENERATED_DIR, f"{vlm_model}_to_{generator_name}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate images
            print(f"Generating images using {generator_name} with descriptions from {vlm_model}...")
            
            # Save generation info
            generation_info = []
            
            for image_name, description in tqdm(limited_descriptions.items()):
                output_path = os.path.join(output_dir, f"generated_{image_name}")
                success = generator.generate_image(description, output_path)
                
                if success:
                    generation_info.append({
                        'original_image': image_name,
                        'generated_image': f"generated_{image_name}",
                        'description': description,
                        'vlm_model': vlm_model,
                        'generator_model': generator_name
                    })
            
            # Save generation info
            info_df = pd.DataFrame(generation_info)
            info_df.to_csv(os.path.join(output_dir, 'generation_info.csv'), index=False)
            
            print(f"Generated {len(generation_info)} images using {generator_name} with descriptions from {vlm_model}")
    
    print("Image generation complete!")

if __name__ == "__main__":
    main() 