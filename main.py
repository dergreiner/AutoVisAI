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

character_prompt= ""
car_prompt = ""
angle_prompt= ""
road_prompt = ""
traffic_prompt= ""
situation_prompt= ""
environment_prompt= ""
position_prompt= ""
shot_description_prompt= ""
negative_prompt= ""
actualprompt= ""

## get selected values
def generate_character_prompt(car_input):
    global character_prompt
    character_prompt = car_input
    return updateprompt()

def generate_car_prompt(car_input):
    global car_prompt
    if car_input == "Define your own car...":
         print("defineyourcar")
         return updateprompt(), gr.Textbox(visible=True)
    else:
        car_prompt = car_input
        print(car_input)
        return updateprompt(), gr.Textbox(visible=False)

def generate_complexcar_prompt(customcar_input):
    global car_prompt
    car_prompt = customcar_input
    return updateprompt()

def generate_angle_prompt(angle_input):
    global angle_prompt
    angle_prompt = angle_input
    return updateprompt()

def generate_road_prompt(road_input):
    global road_prompt
    road_prompt = ", " + road_input
    return updateprompt()

def generate_traffic_prompt(traffic_input):
    global traffic_prompt
    traffic_prompt = ", " + traffic_input
    return updateprompt()

def generate_situation_prompt(situation_input):
    global situation_prompt
    situation_prompt = ", " + situation_input
    return updateprompt()

def generate_environment_prompt(environment_input):
    global environment_prompt
    environment_prompt = ", " + environment_input
    return updateprompt()

def generate_position_prompt(intext_input):
    global position_prompt
    if intext_input == "Inside the car":
        position_prompt = ", inside the car"
    else:
        position_prompt = ", outside the car"
    return updateprompt()

def generate_shot_description_prompt(prompt_input):
    global shot_description_prompt
    shot_description_prompt = ", " + prompt_input
    return updateprompt()

def generate_negative_prompt(negative_prompt_input):
    global negative_prompt
    negative_prompt = negative_prompt_input
    return updateprompt()

## change the complexity of the scene
def changeComplexity(complexity):
        global road_prompt, traffic_prompt, situation_prompt, environment_prompt
        if complexity == "Complex":
            return gr.Column(visible=True)
        else:
            road_prompt = ""
            traffic_prompt = ""
            situation_prompt = ""
            environment_prompt = ""
            return gr.Column(visible=False)

## update the prompt to currently selected values
def updateprompt():
    global actualprompt
    actualprompt= "A " + angle_prompt + " shot" + " of a " + car_prompt + shot_description_prompt + position_prompt  + road_prompt + traffic_prompt + situation_prompt + environment_prompt +" , sketch, monochrome, cinematic, cinematic lightening"
    return actualprompt

def txt2img():
    actualnegativeprompt= negative_prompt + ""
    image = pipeline(prompt = actualprompt, negative_prompt= negative_prompt, width=1064, height=608).images[0]
    print ("[PROMPT]: ", actualprompt)
    print ("[NEGATIVE_PROMPT]: ", actualnegativeprompt)
    return image

with gr.Blocks(title="Storyboard Cars", theme='gstaff/xkcd') as demo:
    gr.Markdown("## Storyboard Cars")
    with gr.Row():
        with gr.Column():
                with gr.Row():
                        with gr.Column():
                            ##character_input = gr.Dropdown(["Anna", "Max"], multiselect=True, label="Character", info="Who is in the scene?")
                            gr.Markdown("## Basic Scene Setup")
                            intext_input =  gr.Radio(["Inside the car", "Outside the car"], label="Scene Positioning", info="Interior or exterior?")
                            car_input = gr.Dropdown(label="Car Type", info="Choose the type of car in the scene", choices=["4-door sedan", "2-door coupe", "Van", "Sports car", "Sports utility", "Pickup truck", "Define your own car..."])
                            customcar_input = gr.Textbox(label="Custom Car", placeholder="your custom car here", visible=False, info="Describe your custom car, be specific")
                            angle_input = gr.Dropdown(["Normal", "Low angle", "High angle", "Close-Up", "Wide"], label="View Angle", info="Choose the View Angle of the scene")
                            simple_input = gr.Radio(["Simple", "Complex"], label="Complexity", info="Choose the complexity of the scene")
                with gr.Row():
                        with gr.Column(visible=False) as detail_cols:
                            gr.Markdown("## Defining the scene")
                            road_input =  gr.Textbox(label="Road", placeholder="bumpy wet road", info="describes the layout of the road, including markings, topology")
                            traffic_input =  gr.Textbox(label="Traffic", placeholder="many other cars", info="defines traffic infrastructures (e.g., traffic signs/lights)")
                            situation_input =  gr.Textbox(label="Objects",placeholder="bumpy wet road", info="describes all the objects, their maneuvers, and interactions in a scenario")
                            Environment_input =  gr.Textbox(label="Environment", placeholder="sunny weather, sun is bright", info="environmental condition like weather and lighting")
                with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Optimizing")
                            prompt_input = gr.Textbox(lines=3, label="Shot Description", placeholder="your prompt here")
                            negative_prompt_input = gr.Textbox(lines=3, label="What do you want to avoid?", placeholder="your negative prompt here")
                            start_button = gr.Button("Generate Image")
        with gr.Column():
            image_output = gr.Image()
            prompt_output = gr.TextArea(value=actualprompt, interactive=False, show_label=False)

    start_button.click(fn=txt2img, inputs=[], outputs=[image_output])

    #dynamic update of the prompt
    ##character_input.change(generate_character_prompt, character_input, prompt_output)
    prompt_input.change(generate_shot_description_prompt, prompt_input, prompt_output)
    car_input.change(generate_car_prompt, car_input, [prompt_output, customcar_input])
    angle_input.change(generate_angle_prompt, angle_input, prompt_output)
    intext_input.change(generate_position_prompt, intext_input, prompt_output)
    negative_prompt_input.change(generate_negative_prompt, negative_prompt_input, prompt_output)
    prompt_input.change(generate_shot_description_prompt, prompt_input, prompt_output)
    road_input.change(generate_road_prompt, road_input, prompt_output)
    traffic_input.change(generate_traffic_prompt, traffic_input, prompt_output)
    situation_input.change(generate_situation_prompt, situation_input, prompt_output)
    Environment_input.change(generate_environment_prompt, Environment_input, prompt_output)
    simple_input.change(changeComplexity, simple_input, detail_cols)
    customcar_input.change(generate_complexcar_prompt, customcar_input, prompt_output)
    
    


demo.launch(share = True, debug = True)