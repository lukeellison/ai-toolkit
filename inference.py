import torch
from diffusers import DiffusionPipeline
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--lora_name", type=str, required=True, help="Name of the lora model"
)
parser.add_argument(
    "-p", "--prompt", type=str, required=True, help="Prompt for the image"
)
parser.add_argument("-s", "--seed", type=int, default=42, help="Seed for the image")
parser.add_argument("-w", "--width", type=int, default=1024, help="Width of the image")
parser.add_argument(
    "-h", "--height", type=int, default=1024, help="Height of the image"
)
parser.add_argument(
    "-S", "--num_steps", type=int, default=50, help="Number of steps for the image"
)
parser.add_argument(
    "-N", "--num_images", type=int, default=1, help="Number of images to generate"
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="output",
    help="Output directory for the images",
)

args = parser.parse_args()
lora_name = args.lora_name

model_id = "black-forest-labs/FLUX.1-dev"
adapter_id = f"output/{lora_name}/{lora_name}.safetensors"
pipeline = DiffusionPipeline.from_pretrained(model_id)
pipeline.load_lora_weights(adapter_id)

pipeline.to(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
images = pipeline(
    prompt=args.prompt,
    num_inference_steps=args.num_steps,
    generator=torch.Generator(
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    ).manual_seed(args.seed),
    width=args.width,
    height=args.height,
    num_images_per_prompt=args.num_images,
).images

# check if output directory exists
if not os.path.exists(args.output):
    os.makedirs(args.output)

for i, image in enumerate(images):
    image.save(f"{args.output}/{i}.png")
