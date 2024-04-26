from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
from torch import autocast
import gradio as gr 
from diffusers.utils import make_image_grid, load_image


mycss = """
.black-text {
    color: black !important;
}
.optional-details {
    color: black !important;
    background-color: LightGrey;
}
.red-text {
    text-decoration-line: underline;
    text-decoration-color: red;
}
"""

model_id1 = "runwayml/stable-diffusion-v1-5"
model_id2 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id3 = "stabilityai/stable-diffusion-2"
model_id4 = "SG161222/Realistic_Vision_V5.1_noVAE"
model_id5 = "stabilityai/stable-diffusion-xl-base-1.0"

torch.cuda.empty_cache()

pipeline = AutoPipelineForText2Image.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipeline.to("cuda")
pipelineImg2Img= AutoPipelineForImage2Image.from_pipe(pipeline)
pipelineImg2Img.to("cuda")

##pipelineImg2Img.enable_model_cpu_offload()

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

## define the options for the characters and cars
character_options = {"Alexa": {"look alike":"looks like Emma Watson", "clothes": "wearing a red dress", "age": "32 years old", "height": "tall", "weight": "normal weight", "details": "gold necklace around her neck", "prompt": "a tall normal weight 32 years old person(looks like Emma Watson) wearing a red dress and gold necklace around her neck"}} 
car_options = {"Mercedes S-Class": {"model": "Merdeces S-Class","exterior": "futuristic", "interior": "calm, clean, modern", "details": "big windows", "prompt": "a futuristic Mercedes S-Class with a calm, clean, modern interior"}}



## generate the image , guidance_scale=7.5,  width=1064, height=608
def generateimage(init_image, strength_slider):
    url = "https://global.discourse-cdn.com/hellohellohello/original/3X/c/1/c1f3e18330ce1273809da41d16a634d7becdd420.jpeg"
    
    actualnegativeprompt= negative_prompt + ""
    if init_image is not None:
        image = pipelineImg2Img(prompt = actualprompt, image=init_image, strength=strength_slider, guidance_scale=10.5).images[0]
        print("[Model]: Img2Img")
        print("Strength: " + str(strength_slider))
    else:
        image = pipeline(prompt = actualprompt, negative_prompt= negative_prompt, width=1064, height=608).images[0]
        print("[Model]: Text2Img")

    print ("[PROMPT]: ", actualprompt)
    print ("[NEGATIVE_PROMPT]: ", actualnegativeprompt)
    return image


## add new characters and cars

def addcharacter(character_name_input, character_lookalike_input, character_clothes_input, character_age_input, character_height_input, character_weight_input, character_details_input):
        global character_options, character_choices
        ## get values
        age = str(character_age_input) + " years old"
        clothes = "wearing " + character_clothes_input
        lookalike = "looks like " + character_lookalike_input
        if character_height_input is None:
            height = ""
            print("height is none")
        else:
            height = str(character_height_input) + " "

        if character_weight_input is None:
            weight = ""
        else:
            weight = str(character_weight_input)+ " "
        if character_details_input == "":
            details = ""
        else:
            details = " and " + character_details_input
        character_prompt = "a " +  height + weight + age + " person(" + lookalike + ") " + clothes + details

        ## store values
        character_details = {
            "look alike": lookalike,
            "clothes": clothes,
            "age": age,
            "height": height,
            "weight": weight,
            "details": details,
            "prompt": character_prompt
        }
        character_options[character_name_input] = character_details
        print("added character")
        return update_characters_outputlist(character_options)

def update_characters_outputlist(character_options):
    all_characters = []
    for character_name, details in character_options.items():
        # Zugriff auf das spezifische Detail "prompt"
        specific_detail = details.get("prompt", "N/A") # "N/A" als Standardwert, falls "prompt" nicht vorhanden ist
        all_characters.append(f"{character_name}: {specific_detail}")
    
    return "\n".join(all_characters)

## a futuristic Mercedes S-Class with a calm, clean, modern interior
def addcar(car_name_input, car_choice_input, car_exterior_description, car_interior_description, car_details_input):
        global car_options
        car_model = car_choice_input
        if car_exterior_description == "":
            car_exterior = ""
        else:
            car_exterior = car_exterior_description + " "
        if car_interior_description == "":
            car_interior = ""
        else:
            car_interior = " with a " + car_interior_description + " interior"
        if car_details_input == "":
            details = ""
        else:
            details = " and " + car_details_input

        car_prompt ="a " + car_exterior+ car_choice_input + car_interior + details

        ## store values
        car_details= {
            "model": car_model,
            "exterior": car_exterior,
            "interior": car_interior,
            "details": details,
            "prompt": car_prompt
        }
        car_options[car_name_input] = car_details
        print("added car")
        return update_car_outputlist(car_options)

def update_car_outputlist(car_options):
    all_cars = []
    for car_name, prompt in car_options.items():
        # Zugriff auf das spezifische Detail "prompt"
        specific_detail = prompt.get("prompt", "N/A") # "N/A" als Standardwert, falls "prompt" nicht vorhanden ist
        all_cars.append(f"{car_name}: {specific_detail}")
    
    return "\n".join(all_cars)


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
    actualprompt= "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    ## actualprompt= "A " + angle_prompt + " shot" + " of a " + car_prompt + shot_description_prompt + position_prompt  + road_prompt + traffic_prompt + situation_prompt + environment_prompt +" , sketch, monochrome, cinematic, cinematic light"
    return actualprompt

## show the tabs
def showmenu():
    return gr.Row(visible=True), gr.Button(visible=False), gr.TextArea(visible=False)



with gr.Blocks(title="Storyboard Cars", theme="gstaff/xkcd@=0.0.4", css=mycss) as demo:
    gr.Markdown("## Storyboard Cars")
    with gr.Column():
         infotext = gr.TextArea(value="Welcome to Storyboard for Cars, \n\nYou will generate new frames throughout the program. \n\nEvery time your frame is finished, it will upload to the overview tab. \nThe generation of frames may take some time depending on the performance of the system. Therefore we ask for your patience.\n\nThank you.", interactive=False, show_label=False)
         start_button = gr.Button("Start")
    with gr.Row(visible = False) as generating:
         with gr.Tab(label = "1. Defining Basics",):
            gr.Markdown("Here you can define options that are generally applicable to all scenes.")
            with gr.Row():
                    with gr.Column():
                        character_choices = gr.TextArea(update_characters_outputlist(character_options), label="Personas", info="Available personas to choose from")
                        with gr.Accordion("Click to create your own Persona", open=False, elem_classes=["optional-details"]):
                            character_name_input = gr.Textbox(label="Persona Name", placeholder="your persona name here", info="Name your persona for later reference")
                            character_lookalike_input = gr.Textbox(label="Well-known Lookalike", placeholder="your character look alike here", info="To get consistent results, please provide a lookalike of the character, you desire. It should be a celebrity or a well-known person.")
                            character_clothes_input = gr.Textbox(label="Clothing", placeholder=" a red dress", info="What is your Persona wearing?")
                            with gr.Accordion("Optional details", open=False, elem_classes=["black-text"]):
                                 character_age_input = gr.Slider(1, 100, step=1, value=32, interactive=True, label="Age", info="Choose an age between 1 and 100")
                                 character_height_input = gr.Dropdown(label="Height", info="Choose heigth", choices=["very short", "short", "normal sized", "tall", "very tall"])
                                 character_weight_input = gr.Dropdown(label="Weight", info="Choose the weight", choices=["very thin", "thin", "normal weigth", "overweight", "obese"])
                                 character_details_input = gr.Textbox(label="Additional details", placeholder="big face tatoo", info="Something Special about your character?")
                            savecharacter_btn = gr.Button("Save Persona")
            with gr.Row():
                    with gr.Column():
                        car_choices = gr.TextArea(update_car_outputlist(car_options), label="Cars", info="Available cars to choose from")
                        with gr.Accordion("Click to create your own Car", open=False, elem_classes=["optional-details"]):
                            car_name_input = gr.Textbox(label="Car Name", placeholder="your car name here", info="Name your custom car for later reference")
                            car_choice_input =  gr.Textbox(placeholder="Based on a real world vehicle or pick a vehicle class(4-door sedan, 2-door coupe, Van/wagon, Sports car, Sports utility, Pickup truck) or define your own", label="Vehicle", info="What type of vehicle")
                            with gr.Accordion("Optional details", open=False, elem_classes=["black-text"]):
                                car_exterior_description = gr.Textbox(label="Exterior Describtion", placeholder="futuristic, historic, autonomous", info="Describe your custom car, be specific")
                                car_interior_description = gr.Textbox(label="Interior Describtion", placeholder="calm clean intuitive environment", info="Describe your custom car, be specific")
                                car_details_input = gr.Textbox(label="Additional details", placeholder="big windows", info="Something special about your car?")
                            savecar_btn = gr.Button("Save Car")
         with gr.Tab(label = "2. Generate Image") as tab2:
            with gr.Row():
                with gr.Column():
                        with gr.Row():
                                with gr.Column():
                                    ##character_input = gr.Dropdown(["Anna", "Max"], multiselect=True, label="Character", info="Who is in the scene?")
                                    gr.Markdown("## Basic Scene Setup")
                                    img_input = gr.Image()
                                    strength_slider = gr.Slider(label="Strength",maximum = 1, value = 0.75 , step = 0.10, info="The strength of the img2img effect")
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
            gallery = gr.Gallery(label="Generated images", show_label=False)
            image2_output = gr.Image()
    start_button.click(fn=showmenu, inputs=[], outputs=[generating, start_button, infotext])
    generate_button.click(fn=generateimage, inputs=[img_input, strength_slider], outputs=[image_output])

    
        
    
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
    savecharacter_btn.click(addcharacter, inputs=[character_name_input, character_lookalike_input, character_clothes_input, character_age_input, character_height_input, character_weight_input, character_details_input], outputs=character_choices)
    savecar_btn.click(addcar, inputs=[car_name_input, car_choice_input, car_exterior_description, car_interior_description, car_details_input], outputs=car_choices)
    


demo.launch(share = True, debug = True)