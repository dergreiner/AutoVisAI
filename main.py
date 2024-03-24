from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
import gradio as gr 


model_id1 = "runwayml/stable-diffusion-v1-5"
model_id2 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id3 = "stabilityai/stable-diffusion-2"


pipeline = DiffusionPipeline.from_pretrained(model_id3, torch_dtype=torch.float16)
pipeline.to("cuda")


pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)


prompt= "young man sitting in a red mercedes sports car, car is driving on the road, cinematic"
negative_prompt= "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, anime:1.4) text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorlydrawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, badproportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms,missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, nsfw, inclined head, tilted head, two persons, text, symbol, logo, artist signature" 


def txt2img(prompt, negative_prompt):
    image = pipeline(prompt, negative_prompt= negative_prompt, num_inference_steps=20, width=640, height=360).images[0]
    print ("[PROMPT]: ", prompt)
    return image


txt2img(prompt, negative_prompt)



with gr.Blocks(title="Storyboard Cars", theme='gstaff/xkcd') as demo:
    gr.Markdown("## Storyboard Cars")
    with gr.Row():
        with gr.Column():
            gr.Dropdown(["Anna", "Max"], multiselect=True, label="Character", info="Who is in the scene?")
            gr.Dropdown(["4-door sedan", "2-door coupe", "Van/wagon", "Sports car", "Sports utility", "Pickup truck"], label="Car Type", info="Choose the type of car in the scene"),
            prompt_input = gr.Textbox(lines=3, label="Prompt", placeholder="your prompt here")
            negative_prompt_input = gr.Textbox(lines=3, label="Negative Prompt", placeholder="your negative prompt here")
            checkbox_input =  gr.CheckboxGroup(["Inside the car", "Outside the car"], label="Scene Positioning", info="Where is the scene taking place?")
            
            start_button = gr.Button("Generate Image")
        with gr.Column():
            image_output = gr.Image()
   
    start_button.click(fn=txt2img, inputs=[prompt_input, negative_prompt_input], outputs=[image_output])

demo.launch(share = True, debug = True)