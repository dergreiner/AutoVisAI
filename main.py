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
    url = "https://global.discourse-cdn.com/hellohellohello/original/3X/c/1/c1f3e18330ce1273809da41d16a634d7becdd420.jpeg"
    
    actualnegativeprompt= negative_prompt + ""
    if init_image is not None:
        image = pipelineImg2Img(prompt = actualprompt, image=init_image, strength=strength_slider, guidance_scale=10.5).images[0]
        print("[Model]: Img2Img")
        print("Strength: " + str(strength_slider))
    else:
        image = pipeline(prompt = actualprompt, negative_prompt= negative_prompt, width=1064, height=608).images[0]
        print("[Model]: Text2Img")

    allimages.append((image, actualprompt))
    print ("[PROMPT]: ", actualprompt)
    print ("[NEGATIVE_PROMPT]: ", actualnegativeprompt)
    return image, allimages


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
        character_emotions_prompt =  character_emotions_input
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

## change the view
def changeView(complexity):
        global view_selection, actualprompt, character_select_prompt
        actualprompt= ""
        character_select_prompt = ""
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

    if view_selection == "Exterior of the vehicle":
        
        actualprompt= "A " + angle_prompt + " monochrome minimalistic sketch" + " of " + car_options[car_select_prompt]["exterior"] +" " + car_options[car_select_prompt]["model"] + details_car_prompt + " "+  ext_action_prompt + ext_scene_prompt + ext_road_prompt + ext_environment_prompt + ext_traffic_prompt + ", "+ character_select_prompt + character_action_prompt + character_emotions_prompt + ext_details_prompt + else_prompt + " , sketch, monochrome, cinematic, cinematic light"
    elif view_selection == "Interior of the vehicle":
        
        actualprompt= "A " + angle_prompt + " monochrome minimalistic sketch" + " of a " + character_emotions_prompt + character_select_prompt + character_action_prompt + "a " + car_options[car_select_prompt]["model"] + " with a "+ car_options[car_select_prompt]["interior"]+ " interior"  + int_character_sitting_prompt + int_device_describition_prompt + int_details_prompt + else_prompt + " , sketch, monochrome, cinematic, cinematic light"
    else:
        actualprompt= else_prompt 
        ##actualprompt= "a monochrome minimalistic sketch, " + else_prompt + ", cinematic, cinematic light"
    return actualprompt

## show the tabs
def showmenu():
    return gr.Row(visible=True), gr.Button(visible=False), gr.TextArea(visible=False)



with gr.Blocks(title="Storyboard Cars", theme="gstaff/xkcd@=0.0.4", css=mycss) as demo:
    gr.Markdown("## Storyboard Cars")
    with gr.Column():
         infotext = gr.TextArea(value="Welcome to Storyboard for Cars, \n\nYou will generate new frames throughout the program. \n\nEvery time your frame is finished, it will upload to the overview tab. \nThe generation of frames may take some time depending on the performance of the system. Therefore we ask for your patience.\n\nThank you.", interactive=False, show_label=False)
         start_button = gr.Button("Start", elem_classes=["button"])
    with gr.Row(visible = False) as generating:
         with gr.Tab(label = "1. Defining Basics",):
            gr.Markdown("Here you can define options that are generally applicable to all scenes.")
            with gr.Row():
                    with gr.Column():
                        character_choices = gr.TextArea(update_characters_outputlist(character_options), label="Personas", info="Available personas to choose from")
                        with gr.Accordion("Click to create your own Persona", open=False, elem_classes=["accent-grey"]):
                            character_name_input = gr.Textbox(label="Persona Name*", placeholder="your persona name here", info="Name your persona for later reference", elem_id="redtext")
                            character_lookalike_input = gr.Textbox(label="Well-known Lookalike*", placeholder="your character look alike here", info="To get consistent results, please provide a lookalike of the character, you desire. It should be a celebrity or a well-known person.")
                            character_clothes_input = gr.Textbox(label="Clothing*", placeholder=" a red dress", info="What is your Persona wearing?")
                            with gr.Accordion("Optional details", open=False, elem_classes=["optional-details"]):
                                 character_age_input = gr.Slider(1, 100, step=1, value=32, interactive=True, label="Age", info="Choose an age between 1 and 100", elem_classes=["optional-details"])
                                 character_height_input = gr.Dropdown(label="Height", info="Choose heigth", choices=["very short", "short", "normal sized", "tall", "very tall"], elem_classes=["optional-details"])
                                 character_weight_input = gr.Dropdown(label="Weight", info="Choose the weight", choices=["very thin", "thin", "normal weigth", "overweight", "obese"], elem_classes=["optional-details"])
                                 character_details_input = gr.Textbox(label="Additional details", placeholder="big face tatoo", info="Something Special about your character?", elem_classes=["optional-details"])
                            savecharacter_btn = gr.Button("Save Persona", elem_classes=["button"])
            with gr.Row():
                    with gr.Column():
                        car_choices = gr.TextArea(update_car_outputlist(car_options), label="Cars", info="Available cars to choose from")
                        with gr.Accordion("Click to create your own Car", open=False, elem_classes=["accent-grey"]):
                            car_name_input = gr.Textbox(label="Car Name*", placeholder="your car name here", info="Name your custom car for later reference")
                            car_choice_input =  gr.Textbox(placeholder="Based on a real world vehicle or pick a vehicle class(4-door sedan, 2-door coupe, Van/wagon, Sports car, Sports utility, Pickup truck) or define your own", label="Vehicle*", info="What type of vehicle")
                            with gr.Accordion("Optional details", open=False, elem_classes=["optional-details"]):
                                car_exterior_description = gr.Textbox(label="Exterior Describtion", placeholder="futuristic, historic, autonomous", info="Describe your custom car, be specific")
                                car_interior_description = gr.Textbox(label="Interior Describtion", placeholder="calm clean intuitive environment", info="Describe your custom car, be specific")
                                car_details_input = gr.Textbox(label="Additional details", placeholder="big windows", info="Something special about your car?")
                            savecar_btn = gr.Button("Save Car", elem_classes=["button"])
         with gr.Tab(label = "2. Generate Image") as tab2:
            with gr.Row():
                with gr.Column():
                        with gr.Row():
                                with gr.Column():
                                    ##character_input = gr.Dropdown(["Anna", "Max"], multiselect=True, label="Character", info="Who is in the scene?")
                                    gr.Markdown("## Basic Scene Setup")
                                    car_input = gr.Dropdown(label="Car Type", info="Choose the type of car in the scene", choices=car_options)
                                    simple_input = gr.Radio(["Exterior of the vehicle", "Interior of the vehicle", "Else"], label="Scene Positioning", info="Interior or exterior?")
                                   
                        with gr.Row():
                                with gr.Column(visible=False) as ext_cols:
                                    with gr.Accordion("Exterior", open=True, elem_classes=["accent-grey"]):
                                        action_input = gr.Textbox(label="Action*", placeholder="driving", info="describes the action of the scene")
                                        scene_input = gr.Textbox(label="Scene*", placeholder="forest", info="describes the setting of the scene")
                                        with gr.Accordion("Optional - more details", open=False, elem_classes=["optional-details"]):
                                            road_input =  gr.Textbox(label="Road", placeholder="bumpy wet road", info="describes the layout of the road, including markings, topology")
                                            Environment_input =  gr.Textbox(label="Environment", placeholder="sunny weather, sun is bright", info="environmental conditions like weather and lighting")
                                            traffic_input =  gr.Textbox(label="Traffic", placeholder="many other cars", info="Additional things that are happening? (traffic, other objects and interactions)")
                                            with gr.Accordion("Is one of your personas visible?", open=False, elem_classes=["black-text"]):
                                                character_select_ext = gr.Dropdown(label="Persona", info="Choose the person in the scene", choices=character_options)
                                                character_action_input_ext = gr.Textbox(label="Action", placeholder="walking", info="describes the action of the person")
                                                character_emotions_input_ext = gr.Textbox(label="Emotions", placeholder="happy", info="describes the emotions of the person")
                                            ext_details_input = gr.Textbox(label="Additional details", placeholder=" ", info="Something else?")
                                with gr.Column(visible=False) as int_cols:
                                    with gr.Accordion("Interior", open=True, elem_classes=["accent-grey"]):
                                        int_focus_select = gr.Radio(["Character Interaction", "Technical Device"], label="What is the primary point of interest?*", info="you only want to map a technical device in the car?")
                                        with gr.Accordion("Character interaction", open=True, visible=False) as character_accordion:
                                            character_select_int = gr.Dropdown(label="Persona*", interactive=True, info="Choose the person in the scene", choices=character_options)
                                            character_action_input_int = gr.Textbox(label="Action*", placeholder="sitting in the drivers seat", info="describes the action of the person")
                                            character_emotions_input_int = gr.Textbox(label="Emotions", placeholder="angry/happy/sad/excited", info="describes the emotions of the person. the person is ...")
                                                
                                        with gr.Accordion("Technical Device", open=True, visible=False) as device_accordion:
                                             device_describition_input = gr.Textbox(label="Describe the Device*", placeholder=" futuristic dashboard ", info="Describe the Device (HUD, Dashboard, Steering Wheel, Windshield, Center Stack, Handheld, wearable, Brake)", elem_classes=["required-details"])
                                        int_details_input = gr.Textbox(label="Additional details", placeholder=" ", info="Something else?")
                                with gr.Column(visible=False) as else_cols:
                                    with gr.Accordion("Else", open=True, elem_classes=["accent-grey"]) as else_accordion:
                                        else_prompt_input =  gr.Textbox(label="Scene Describtion*", placeholder="Paris at night", info="Give a description, what you want to visualize!")
                                    
                        with gr.Row():
                                with gr.Column():
                                    with gr.Accordion("Finetuning", open=False, elem_classes=["accent-grey"]):
                                        with gr.Accordion("Reference picture?", open=False, elem_classes=["optional-details"]):
                                            img_input = gr.Image()
                                            strength_slider = gr.Slider(label="Strength",maximum = 1, value = 0.75 , step = 0.10, info="The strength of the img2img effect")
                                        angle_input = gr.Dropdown(["Normal", "Low angle", "High angle", "Close-Up", "Wide"], label="View Angle", info="Choose the View Angle of the scene")
                                        negative_prompt_input = gr.Textbox(lines=3, label="What do you want to avoid?", placeholder="your negative prompt here")
                                    generate_button = gr.Button("Generate Image", elem_classes=["button"])
                with gr.Column():
                    image_output = gr.Image()
                    prompt_output = gr.TextArea(value=actualprompt, interactive=False, show_label=False)
         with gr.Tab(label = "Storyboard Overview") as tab3:
            gr.Markdown("## 3. Storyboard Overview")
            gallery = gr.Gallery(label="Generated images", show_label=True, elem_id="gallery", columns=[3], rows=[1], object_fit="contain", height="auto")
            
    

    
        
    #functions for the dropdowns
    simple_input.change(changeView, simple_input, outputs=[ext_cols, int_cols, else_cols])
    int_focus_select.change(changeInt, int_focus_select, outputs=[character_accordion, device_accordion])

    # Finetuining
    negative_prompt_input.change(generate_negative_prompt, negative_prompt_input, prompt_output)
    angle_input.change(generate_angle_prompt, angle_input, prompt_output)
    

    ##character_input.change(generate_character_prompt, character_input, prompt_output)
   

    # select the car
    car_input.change(generate_car_prompt, car_input, [prompt_output])

    #exterior car
    scene_input.change(generate_scene_prompt, scene_input, prompt_output)
    action_input.change(generate_car_action_prompt, action_input, prompt_output)
    
    road_input.change(generate_road_prompt, road_input, prompt_output)
    Environment_input.change(generate_environment_prompt, Environment_input, prompt_output)
    traffic_input.change(generate_traffic_prompt, traffic_input, prompt_output)
    character_select_ext.change(generate_character_prompt, character_select_ext, prompt_output)
    character_action_input_ext.change(generate_action_prompt, character_action_input_ext, prompt_output)
    character_emotions_input_ext.change(generate_emotions_prompt, character_emotions_input_ext, prompt_output)
    ext_details_input.change(generate_ext_details_prompt, ext_details_input, prompt_output)

    # interior car
    character_select_int.change(generate_character_prompt, character_select_int, prompt_output)
    character_action_input_int.change(generate_action_prompt, character_action_input_int, prompt_output)
    character_emotions_input_int.change(generate_emotions_prompt, character_emotions_input_int, prompt_output)
    
    device_describition_input.change(generate_device_prompt, device_describition_input, prompt_output)
    int_details_input.change(generate_int_details_prompt, int_details_input, prompt_output)

    # else
    else_prompt_input.change(generate_else_prompt, else_prompt_input, prompt_output)
                                   
    # Buttons
    savecharacter_btn.click(addcharacter, inputs=[character_name_input, character_lookalike_input, character_clothes_input, character_age_input, character_height_input, character_weight_input, character_details_input], outputs=[character_choices, character_select_ext, character_select_int])
    savecar_btn.click(addcar, inputs=[car_name_input, car_choice_input, car_exterior_description, car_interior_description, car_details_input], outputs=[car_choices, car_input])
    start_button.click(fn=showmenu, inputs=[], outputs=[generating, start_button, infotext])
    generate_button.click(fn=generateimage, inputs=[img_input, strength_slider], outputs=[image_output, gallery])


demo.launch(share = True, debug = True)