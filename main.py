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
character_options = {"Alexa": ["18 years old" , "looks like Emma Watson"]}
car_options = [("4-door sedan", "4-door sedan"), ("2-door coupe","2-door coupe"),("Van", "Van"), ("Sports car", "Sports car"), ("Sports utility", "Sports utility"), ("Pickup truck", "Pickup truck")]
character_names =list(character_options.keys())


## add new characters and cars

def addcharacter(character_name_input,character_age_input, character_lookalike_input):
        global character_options, character_names, character_choices
        age = str(character_age_input) + "years old"
        lookalike = "looks like " + character_lookalike_input
        character_options[character_name_input] = [age, lookalike]
        print("added character")
        character_names =list(character_options.keys())
        return character_options
    
def addcar(car_name_input,car_description):
        global car_options, car_input
        car_options.append((car_name_input, car_description))
        print("added car")
        print(car_input.choices)
        car_input.choices = car_options
        print(car_input.choices)
        return car_options, gr.Dropdown(choices=car_options)


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

## show the tabs
def showmenu():
    return gr.Row(visible=True), gr.Button(visible=False), gr.TextArea(visible=False)

## generate the image
def txt2img():
    actualnegativeprompt= negative_prompt + ""
    image = pipeline(prompt = actualprompt, negative_prompt= negative_prompt, width=1064, height=608).images[0]
    print ("[PROMPT]: ", actualprompt)
    print ("[NEGATIVE_PROMPT]: ", actualnegativeprompt)
    return image

def img2img():
    pipeline2= AutoPipelineForImage2Image.from_pipeline(pipeline)
    image = pipeline(images = image, width=1064, height=608).images[0]
    return image

with gr.Blocks(title="Storyboard Cars", theme="gstaff/xkcd@=0.0.4") as demo:
    gr.Markdown("## Storyboard Cars")
    with gr.Column():
         infotext = gr.TextArea(value="Welcome to Storyboard for Cars, \n\nYou will generate new frames throughout the program. \n\nEvery time your frame is finished, it will upload to the overview tab. \nThe generation of frames may take some time depending on the performance of the system. Therefore we ask for your patience.\n\nThank you.", interactive=False, show_label=False)
         start_button = gr.Button("Start")
    with gr.Row(visible = False) as generating:
         with gr.Tab(label = "1. Defining Basics",):
            gr.Markdown("Here you can define options that are generally applicable to all scenes.")
            with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Character")
                        character_choices = gr.TextArea(character_options, label="Characters", info="Available characters to choose from")
                        with gr.Accordion("Click to create your own Character", open=False):
                            character_name_input = gr.Textbox(label="Character Name", placeholder="your character name here", info="Name your character for later reference")
                            character_age_input = gr.Slider(1, 100, step=1, value=32, interactive=True, label="Age", info="Choose an age between 1 and 100")
                            character_lookalike_input = gr.Textbox(label="Character Lookalike", placeholder="your character look alike here", info="To get consistent results, please provide a lookalike of the character, you desire. It should be a celebrity or a well-known person.")
                            savecharacter_btn = gr.Button("Save Character")
            with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Car")
                        car_choices = gr.TextArea(car_options, label="Cars", info="Available cars to choose from")
                        with gr.Accordion("Click to create your own Car", open=False):
                            car_name_input = gr.Textbox(label="Car Name", placeholder="your car name here", info="Name your custom car for later reference")
                            car_description = gr.Textbox(label="Car Describtion", placeholder="your car here", info="Describe your custom car, be specific")
                            savecar_btn = gr.Button("Save Car")
         with gr.Tab(label = "2. Generate Image") as tab2:
            with gr.Row():
                with gr.Column():
                        with gr.Row():
                                with gr.Column():
                                    ##character_input = gr.Dropdown(["Anna", "Max"], multiselect=True, label="Character", info="Who is in the scene?")
                                    gr.Markdown("## Basic Scene Setup")
                                    intext_input =  gr.Radio(["Inside the car", "Outside the car"], label="Scene Positioning", info="Interior or exterior?")
                                    car_input = gr.Dropdown(label="Car Type", info="Choose the type of car in the scene", choices=car_options)
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
                                    generate_button = gr.Button("Generate Image")
                with gr.Column():
                    image_output = gr.Image()
                    prompt_output = gr.TextArea(value=actualprompt, interactive=False, show_label=False)
         with gr.Tab(label = "Storyboard Overview") as tab3:
            gr.Markdown("## 3. Storyboard Overview")
            image2_output = gr.Image()
    start_button.click(fn=showmenu, inputs=[], outputs=[generating, start_button, infotext])
    generate_button.click(fn=txt2img, inputs=[], outputs=[image_output])

    
        
    
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
    savecharacter_btn.click(addcharacter, inputs=[character_name_input,character_age_input, character_lookalike_input], outputs=character_choices)
    savecar_btn.click(addcar, inputs=[car_name_input,car_description], outputs=[car_choices, car_input])
    


demo.launch(share = True, debug = True)