import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import DigitGenerator

# Load model
@st.cache_resource
def load_model():
    model = DigitGenerator()
    model.load_state_dict(torch.load("model/digit_generator.pth", map_location="cpu"))
    model.eval()
    return model

# Generate 5 images
def generate_images(model, digit, noise_dim=64):
    noise = torch.randn(5, noise_dim)
    labels = torch.tensor([digit]*5)
    with torch.no_grad():
        imgs = model(noise, labels).numpy()
    return imgs

def show_images(imgs):
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for ax, img in zip(axes, imgs):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.axis("off")
    st.pyplot(fig)

# Streamlit UI
st.title("Handwritten Digit Generator (0–9)")
digit = st.selectbox("Choose a digit to generate", list(range(10)))
if st.button("Generate Images"):
    model = load_model()
    imgs = generate_images(model, digit)
    show_images(imgs)