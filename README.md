# Testing StableDiffusion 1.4 for my master's thesis

This repo uses Stable Diffusion 1.4 for text-to-image generation. It does not include layout guidance or bounding box constraints — images are generated freely based on the prompt.

## Quick start

```bash
conda create --name sd python=3.10
conda activate sd
pip install -r requirements.txt
```

## Image generation

The .csv file containing the prompts should be inside a folder named `prompts` that is posiotioned in the root of the project.
