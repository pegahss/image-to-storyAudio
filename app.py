import streamlit as st
from io import BytesIO
import tempfile
from PIL import Image
import os
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv(find_dotenv())

HUGGINGFACEHUB_API_TOKEN = os.getenv("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")

# Function for image-to-text


def img2text(image_path):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-large")

    text = image_to_text(image_path)[0]['generated_text']

    return text

# Function for generating a story


def generate_story(senario):
    template = """
    you are a story teller;
    you can generate a short story based on a simple narrative, the story should be no more than 40 words;
    
    CONTEXT: {senario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["senario"])

    story_llm = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo",
        temperature=1,
        openai_api_key="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"
    ), prompt=prompt, verbose=True)

    story = story_llm.predict(senario=senario)

    return story

# Function for text-to-speech


def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    audio = response.content

    return audio

# Streamlit App


def main():
    st.title("Image-to-Story-to-Speech App")

    uploaded_image = st.file_uploader(
        "Upload an image...", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image",
                 use_column_width=True)
        st.write("Generating story from image...")

        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
            temp_image.write(uploaded_image.read())
            temp_image_path = temp_image.name

        senario = img2text(temp_image_path)
        story = generate_story(senario)

        st.write("Generated Story:")
        st.write(story)

        st.write("Converting story to speech...")
        audio = text2speech(story)
        st.audio(audio, format="audio/flac")


if __name__ == "__main__":
    main()
