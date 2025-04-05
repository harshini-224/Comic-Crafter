import streamlit as st
from PIL import Image
import torch
import random
import os
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_models():
    """
    Loads Stable Diffusion model for images and GPT-2 model for story generation.
    """
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
)

    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use GPT-2 instead of Mistral
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return pipe, tokenizer, model

pipe, tokenizer, model = load_models()

def generate_story(theme):
    """
    Uses GPT-2 to generate a four-panel Studio Ghibli-style story.
    """
    prompt = f"Write a four-panel story in the style of a Studio Ghibli film about {theme}."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story.split(". ")[:4]  # Ensure only four panels

def generate_image(prompt, filename):
    """
    Uses Stable Diffusion to generate an image with a Ghibli-like style.
    """
    stylized_prompt = f"Studio Ghibli style, soft lighting, whimsical fantasy, {prompt}"
    image = pipe(stylized_prompt).images[0]
    image.save(filename)

def create_comic(panel_images, output_filename="comic_strip.png"):
    """
    Combines four images into a single horizontal comic strip.
    """
    images = [Image.open(img) for img in panel_images]
    min_height = min(img.height for img in images)
    images = [img.resize((int(img.width * min_height / img.height), min_height)) for img in images]
    total_width = sum(img.width for img in images)
    comic_strip = Image.new("RGB", (total_width, min_height))
    x_offset = 0
    for img in images:
        comic_strip.paste(img, (x_offset, 0))
        x_offset += img.width
    comic_strip.save(output_filename)
    return output_filename

# Streamlit UI
st.title("AI-Powered Comic Crafter System")
theme = st.text_input("Enter a theme for your comic story:")
if st.button("Generate Comic"):
    if theme:
        st.write("Generating story...")
        prompts = generate_story(theme)
        panel_filenames = [f"panel{i+1}.png" for i in range(4)]
        
        for prompt, filename in zip(prompts, panel_filenames):
            generate_image(prompt, filename)
        
        comic_path = create_comic(panel_filenames)
        st.image(comic_path, caption="Your AI-generated Ghibli-style comic!")
    else:
        st.warning("Please enter a theme.")
