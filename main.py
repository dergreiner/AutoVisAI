from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
import gradio as gr 


model_id1 = "runwayml/stable-diffusion-v1-5"
model_id2 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id3 = "stabilityai/stable-diffusion-2"
model_id4 = "SG161222/Realistic_Vision_V5.1_noVAE"
model_id5 = "stabilityai/stable-diffusion-xl-base-1.0"

torch.cuda.empty_cache()

pipeline = AutoPipelineForText2Image.from_pretrained(model_id3, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipeline.to("cuda")


##pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)


prompt= "young man sitting in a red mercedes sports car, car is driving on the road, cinematic"
negative_prompt= "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, anime:1.4) text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorlydrawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, badproportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms,missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, nsfw, inclined head, tilted head, two persons, text, symbol, logo, artist signature" 
actualprompt= ""

def txt2img(prompt, car_input, angle_input, position_input, negative_prompt):
    actualprompt= "A " + angle_input + " shot" + " of a " + car_input  + prompt + "ketch, monochrome, cinematic, cinematic lightening"
    actualnegativeprompt= negative_prompt + ""
    image = pipeline(prompt = actualprompt, negative_prompt= negative_prompt, num_inference_steps=20).images[0]
    print ("[PROMPT]: ", actualprompt)
    print ("[NEGATIVE_PROMPT]: ", actualnegativeprompt)
    return image

def definecar(car_input):
    car= car_input
    return car


with gr.Blocks(title="Storyboard Cars", theme='gstaff/xkcd') as demo:
    gr.Markdown("## Storyboard Cars")
    with gr.Row():
        with gr.Column():
            gr.Dropdown(["Anna", "Max"], multiselect=True, label="Character", info="Who is in the scene?")
            car_input = gr.Interface(definecar, 
                                        gr.Dropdown(["4-door sedan", "2-door coupe", "Van", "Sports car", "Sports utility", "Pickup truck"], label="Car Type", info="Choose the type of car in the scene"),
                                        gr.Dropdown(["Normal", "Low angle", "High angle", "Close-Up", "Wide"], label="View Angle", info="Choose the View Angle of the scene"),
                                        gr.CheckboxGroup(["Inside the car", "Outside the car"], label="Scene Positioning", info="Where is the scene taking place?"),
                                        gr.Textbox(lines=3, label="Shot Description", placeholder="your describtion here"),
                                        gr.Textbox(lines=3, label="What do you want to avoid?", placeholder="your negative prompt here"),
                                        ##gr.Button("Generate Image",)
                                        outputs="textbox")
           
        
            
            
        with gr.Column():
            image_output = gr.Image()
            prompt_output = gr.TextArea(value="actualprompt", interactive=False, show_label=False)
   
    
    ##start_button.click(fn=txt2img, inputs=[prompt_input, negative_prompt_input], outputs=[image_output])
    

demo.launch(share = True, debug = True)