from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch


model_id1 = "runwayml/stable-diffusion-v1-5"
model_id2 = "dreamlike-art/dreamlike-diffusion-1.0"


pipeline = DiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16)
pipeline.to("cuda")


pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)


prompt= "young man sitting in a red mercedes sports car, car is driving on the road, cinematic"
negative_prompt= "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, anime:1.4) text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorlydrawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, badproportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms,missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, nsfw, inclined head, tilted head, two persons, text, symbol, logo, artist signature, black-white" 

image = pipeline(prompt, negative_prompt= negative_prompt, num_inference_steps=20, width=512, height=512).images[0]

print ("[PROMPT]: ", prompt)
image.show()