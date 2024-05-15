from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
from torch import autocast
import gradio as gr 
from diffusers.utils import make_image_grid, load_image
from PIL import Image 

mycss = """
.font-size {    
    font-size: 20px !important;
}
.black-text {
    color: black !important;
}
.optional-details {
    color: black !important;

}
.accent-grey {
    color: black !important;
    background-color: LightGrey;
}
.required-details {
    color: black !important;
    background-color: PeachPuff;
}
.button {
    color: black !important;
    background-color: Coral;
}
#redtext {
   color: Coral !important;
}
"""
torch.cuda.empty_cache()

model_id1 = "runwayml/stable-diffusion-v1-5"
model_id2 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id3 = "stabilityai/stable-diffusion-2"
model_id4 = "SG161222/Realistic_Vision_V5.1_noVAE"
model_id5 = "stabilityai/stable-diffusion-xl-base-1.0"


pipeline = AutoPipelineForText2Image.from_pretrained(model_id1, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipeline.to("cuda")
pipelineImg2Img= AutoPipelineForImage2Image.from_pipe(pipeline).to("cuda")
pipelineImg2Img.to("cuda")


#exterior
ext_scene_prompt = ""
ext_action_prompt = ""
ext_road_prompt = ""
ext_environment_prompt= ""
ext_traffic_prompt= ""
ext_details_prompt= ""

#character
character_select_prompt= ""
character_action_prompt= ""
character_emotions_prompt= ""

#interior
int_character_sitting_prompt= ""
int_device_describition_prompt= ""
int_details_prompt= ""

#else
else_prompt= ""

# general
view_selection= ""
car_select_prompt= ""
angle_prompt= ""
shot_description_prompt= ""
negative_prompt= ""


actualprompt= ""
allimages = []
## define the options for the characters and cars
character_options = {"Alexa": {"look alike":"looks like Emma Watson", "clothes": "wearing a dress", "age": "32 years old", "height": "tall", "weight": "normal weight", "details": "open hair", "prompt": "tall normal weight 32 years old person(looks like Emma Watson) wearing a dress and open hair"}, None: {"look alike": "looks like", "clothes": "wearing", "age": " years old", "height": "", "weight": "", "details": "", "prompt": ""} }
car_options = {"Mercedes S-Class": {"model": "Merdeces S-Class","exterior": "futuristic", "interior": "calm, clean, modern", "details": "big windows", "prompt": "a futuristic Mercedes S-Class with a calm, clean, modern interior"}}



## generate the image , guidance_scale=7.5,  width=1064, height=608
def generateimage(init_image, strength_slider):
    
    actualnegativeprompt= negative_prompt + ""
    if init_image is not None:
        
        init_image_convert = Image.fromarray(init_image).convert("L")

        init_image_resize = init_image_convert.resize((1064, 608), Image.LANCZOS).convert("RGB")
        image = pipelineImg2Img(prompt = actualprompt, image=init_image_resize, strength=strength_slider, guidance_scale=10.5, width=1064, height=608).images[0]
        print("[Model]: Img2Img")
        print("Strength: " + str(strength_slider))
    else:
        image = pipeline(prompt = actualprompt, negative_prompt= negative_prompt, width=1064, height=608).images[0]
        print("[Model]: Text2Img")

    allimages.append((image, actualprompt))
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
        character_prompt = height + weight + age + " person(" + lookalike + ") " + clothes + details

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
        return update_characters_outputlist(character_options), gr.Dropdown(label="Persona", info="Choose the person in the scene", choices=character_options), gr.Dropdown(label="Persona", info="Choose the person in the scene", choices=character_options)

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
        return update_car_outputlist(car_options), gr.Dropdown(label="Car Type", info="Choose the type of car in the scene", choices=car_options)

def update_car_outputlist(car_options):
    all_cars = []
    for car_name, prompt in car_options.items():
        # Zugriff auf das spezifische Detail "prompt"
        specific_detail = prompt.get("prompt", "N/A") # "N/A" als Standardwert, falls "prompt" nicht vorhanden ist
        all_cars.append(f"{car_name}: {specific_detail}")

    return "\n".join(all_cars)


#exterior set values
def generate_scene_prompt(scene_input):
    global ext_scene_prompt
    if scene_input == "":
        ext_scene_prompt = ""
    else:
        ext_scene_prompt = " in a " + scene_input
    return updateprompt()

def generate_car_action_prompt(action_input):
    global ext_action_prompt
    if action_input == "":
        ext_action_prompt = ""
    else:
        ext_action_prompt = action_input
    return updateprompt()

def generate_road_prompt(road_input):
    global ext_road_prompt
    if road_input == "":
        ext_road_prompt = ""
    else:
        ext_road_prompt = " on a " + road_input + " road"
    return updateprompt()

def generate_environment_prompt(environment_input):
    global ext_environment_prompt
    if environment_input == "":
        ext_environment_prompt = ""
    else:
        ext_environment_prompt = ", " + environment_input
    return updateprompt()

def generate_traffic_prompt(traffic_input):
    global ext_traffic_prompt
    if traffic_input == "":
        ext_traffic_prompt = ""
    else:
        ext_traffic_prompt = ", " + traffic_input
    return updateprompt()

def generate_ext_details_prompt(ext_details_input):
    global ext_details_prompt
    if ext_details_input == "":
        ext_details_prompt = ""
    else:
        ext_details_prompt = ", " + ext_details_input
    return updateprompt()

#character
def generate_character_prompt(character_select):
    global character_select_prompt
    if str(character_select) == "None":
        character_select_prompt = ""
        print("None clause")
        print(character_select_prompt)
    else:
        character_select_prompt = character_options[character_select]["prompt"] + " "
        print(character_select)
    return updateprompt()

def generate_action_prompt(character_action_input):
    global character_action_prompt
    if character_action_input == "":
        character_action_prompt = ""
    else:
        character_action_prompt = str(character_action_input)
    return updateprompt()

def generate_emotions_prompt(character_emotions_input):
    global character_emotions_prompt
    if character_emotions_input == "":
        character_emotions_prompt = ""
    else:
        character_emotions_prompt =  character_emotions_input + " "
    return updateprompt()

# interior set values

def generate_sitting_prompt(character_sitting_input):
    global int_character_sitting_prompt
    if character_sitting_input == "":
        int_character_sitting_prompt = ""
    else:
        int_character_sitting_prompt = character_sitting_input
    return updateprompt()

def generate_device_prompt(device_describition_input):
    global int_device_describition_prompt
    if device_describition_input == "":
        int_device_describition_prompt = ""
    else:
        int_device_describition_prompt = device_describition_input
    return updateprompt()

def generate_int_details_prompt(int_details_input):
    global int_details_prompt
    if int_details_input == "":
        int_details_prompt = ""
    else:
        int_details_prompt = ", " + int_details_input
    return updateprompt()

# else set values
def generate_else_prompt(else_prompt_input):
    global else_prompt
    if else_prompt_input == "":
        else_prompt = ""
    else:
        else_prompt = else_prompt_input
    return updateprompt()

# general set values
def generate_car_prompt(car_input):
    global car_select_prompt, details_car_prompt
    car_select_prompt = car_input

    if car_options[car_select_prompt]["details"] == "":
        details_car_prompt = ""
    else:
        details_car_prompt =" with " + str(car_options[car_select_prompt]["details"])
    return updateprompt()

def generate_angle_prompt(angle_input):
    global angle_prompt
    if angle_input == "":
        angle_prompt = ""
    else:
        angle_prompt = angle_input
    return updateprompt()

def generate_shot_description_prompt(prompt_input):
    global shot_description_prompt
    if prompt_input == "":
        shot_description_prompt = ""
    else:
        shot_description_prompt = ", " + prompt_input
    return updateprompt()

def generate_negative_prompt(negative_prompt_input):
    global negative_prompt
    negative_prompt = negative_prompt_input
    return updateprompt()

def deleteAttributes():
    global ext_scene_prompt, ext_action_prompt, ext_road_prompt, ext_environment_prompt, ext_traffic_prompt, ext_details_prompt, character_select_prompt, character_action_prompt, character_emotions_prompt, int_character_sitting_prompt, int_device_describition_prompt, int_details_prompt, else_prompt, view_selection, car_select_prompt, angle_prompt, shot_description_prompt, negative_prompt, actualprompt
    ext_scene_prompt = ""
    ext_action_prompt = ""
    ext_road_prompt = ""
    ext_environment_prompt = ""
    ext_traffic_prompt = ""
    ext_details_prompt = ""
    character_select_prompt = ""
    character_action_prompt = ""
    character_emotions_prompt = ""
    int_character_sitting_prompt = ""
    int_device_describition_prompt = ""
    int_details_prompt = ""
    else_prompt = ""
    view_selection = ""
    car_select_prompt = ""
    angle_prompt = ""
    shot_description_prompt = ""
    negative_prompt = ""
    actualprompt = ""


## change the view
def changeView(complexity):
        global view_selection, actualprompt
        deleteAttributes()

        if complexity == "Exterior of the vehicle":
            print("exterior")
            view_selection= "Exterior of the vehicle"
            actualprompt= ""
            return gr.Column(visible=True), gr.Column(visible=False) ,gr.Column(visible=False)
        elif complexity == "Interior of the vehicle":
            print("Interior")
            actualprompt= ""
            view_selection= "Interior of the vehicle"
            return gr.Column(visible=False), gr.Column(visible=True) ,gr.Column(visible=False)
        else:
            print("Else")
            view_selection= "Else"
            actualprompt= ""
            return gr.Column(visible=False),  gr.Column(visible=False) ,gr.Column(visible=True)


def changeInt(selection):
        if selection == "Character Interaction":
            return gr.Accordion(visible=True), gr.Accordion(visible=False)
        else:
            print("Device")
            return gr.Accordion(visible=False), gr.Accordion(visible=True)


## update the prompt to currently selected values
def updateprompt():
    global actualprompt
    actualprompt= else_prompt
        ##actualprompt= "a monochrome minimalistic sketch, " + else_prompt + ", cinematic, cinematic light"
    return actualprompt

## show the tabs
def showmenu():
    return gr.Row(visible=True), gr.Button(visible=False), gr.TextArea(visible=False)



with gr.Blocks(title="Storyboard Cars", theme='gradio/monochrome', css=mycss) as demo:
    gr.Markdown("## Storyboard Cars")
    
         
    with gr.Row():
        with gr.Column():
            else_prompt_input =  gr.Textbox(label="Scene Describtion*", placeholder=" ", info="Give a description, what you want to visualize!")  
            with gr.Accordion("Reference picture?", open=False, elem_classes=["optional-details"]):
                    img_input = gr.Image()
                    strength_slider = gr.Slider(label="Strength",maximum = 1, value = 0.75 , step = 0.10, info="The strength of the img2img effect")
            generate_button = gr.Button("Generate Image", elem_classes=["button"])
        with gr.Column():
            image_output = gr.Image()
            prompt_output = gr.TextArea(value=actualprompt, interactive=False, show_label=False)
         



    # else
    else_prompt_input.change(generate_else_prompt, else_prompt_input, prompt_output)

    # Buttons
   
    generate_button.click(fn=generateimage, inputs=[img_input, strength_slider], outputs=[image_output])


demo.launch(share = True, debug = True)