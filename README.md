# Face Recognition and Generation with VLMs

This project implements a pipeline for face feature extraction and image generation using Vision-Language Models (VLMs). The pipeline extracts facial features from images using various VLM models and then generates new images based on these textual descriptions.

## Project Structure

```
face_vlm_project/
├── data/                  # Directory for dataset storage
├── models/                # Directory for model checkpoints (if needed)
├── outputs/               # Directory for output files
│   ├── <vlm_model>/       # Extracted features for each VLM model
│   ├── generated_images/  # Generated images from textual descriptions
│   └── evaluation/        # FID scores and evaluation results
├── scripts/               # Implementation scripts
│   ├── 1_download_dataset.py    # Script to download face dataset
│   ├── 2_extract_features.py    # Extract facial features using VLMs
│   ├── 3_generate_images.py     # Generate images from textual descriptions
│   └── 4_evaluate_fid.py        # Evaluate generated images using FID scores
├── run_pipeline.py        # Main script to run the entire pipeline
└── setup.sh               # Setup script to install dependencies
```

## Setup

### Automatic Setup

The easiest way to set up the project is to use the provided setup script:

```bash
./setup.sh
```

This script will:
1. Create a Python virtual environment
2. Install all required dependencies including pytorch-fid
3. Create necessary project directories

After running the setup script, activate the virtual environment:

```bash
source venv/bin/activate
```

### Manual Setup

Install the required packages:

```bash
pip install -r requirements.txt
pip install pytorch-fid  # For FID score calculation
```

## Usage

### Complete Pipeline

The simplest way to run the entire pipeline is to use the `run_pipeline.py` script:

```bash
python run_pipeline.py --vlm_models paligemma florence cogvlm llama llava --generators stable-diffusion sdxl-turbo --extract_limit 50 --generate_limit 5
```

This will:
1. Download the LFW dataset
2. Extract facial features from 50 images using all specified VLM models
3. Generate images from 5 descriptions using the specified text-to-image models
4. Calculate FID scores between original and generated images

You can skip specific steps using the following flags:
- `--skip_download`: Skip the dataset download step
- `--skip_extraction`: Skip the feature extraction step
- `--skip_generation`: Skip the image generation step
- `--skip_evaluation`: Skip the FID evaluation step

### Individual Steps

Alternatively, you can run each step individually:

#### 1. Download Face Recognition Dataset

```bash
python scripts/1_download_dataset.py --dataset lfw
```

This downloads the LFW (Labeled Faces in the Wild) dataset, which is a medium-sized dataset suitable for face recognition.

#### 2. Extract Facial Features using VLMs

```bash
python scripts/2_extract_features.py --models paligemma florence cogvlm llama llava --limit 100
```

This extracts facial features from the first 100 images in the dataset using all five VLM models:
- PaliGemma
- Florence
- CogVLM
- Llama 3.2-Vision
- LLaVA

The extracted features are saved as JSON and CSV files in the outputs directory.

#### 3. Generate Images from Facial Descriptions

```bash
python scripts/3_generate_images.py --vlm_model all --generators stable-diffusion sdxl-turbo realistic-vision portrait-plus pixart-alpha --limit 10
```

This generates images from the textual descriptions using text-to-image models.

#### 4. Evaluate Generated Images using FID Score

```bash
python scripts/4_evaluate_fid.py --vlm_models all --generators all
```

This calculates the FID (Fréchet Inception Distance) scores between the original images and the generated images for all combinations of VLM models and generators. Lower FID scores indicate better image quality and similarity to the original images.

## Results

The evaluation results are saved in the `outputs/evaluation` directory:
- FID scores for all model combinations are saved in `fid_scores.csv`
- A bar chart of FID scores is saved in `fid_scores.png`

## Models Used

### VLM Models for Feature Extraction
- PaliGemma
- Florence
- CogVLM
- Llama 3.2-Vision
- LLaVA

### Text-to-Image Models for Image Generation
- Stable Diffusion XL
- SDXL Turbo
- Realistic Vision
- Portrait Plus
- PixArt-Alpha

## Notes

- This project requires significant GPU resources for loading and running the VLM and text-to-image models.
- For optimal performance, a GPU with at least 16GB of VRAM is recommended.
- If VRAM is limited, consider using smaller models or processing fewer images.
- The FID evaluation requires the pytorch-fid package, which will be automatically installed if available. 