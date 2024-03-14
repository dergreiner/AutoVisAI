from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch


model_id1 = "runwayml/stable-diffusion-v1-5"
model_id2 = "dreamlike-art/dreamlike-diffusion-1.0"


pipeline = DiffusionPipeline.from_pretrained(model_id2, torch_dtype=torch.float16)
pipeline.to("cuda")


prompt= "line art drawing of a young man sitting in a red mercedes sports car, car is driving on the road, cinematic"
negative_prompt= "ugly"

image = pipeline(prompt, negative_prompt= negative_prompt, num_inference_steps=20, width=512, height=512).images[0]

print ("[PROMPT]: ", prompt)
image.show()