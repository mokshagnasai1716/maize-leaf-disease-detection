import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from torch import nn

# Load Model Function
def load_model(model_path, device):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Prediction Function
def predict_image(image, model, transform, device, labels_for_viz):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return labels_for_viz[predicted.item()]

# Streamlit App
def main():
    st.title("Plant Disease Classification")
    st.write("Upload an image of a plant leaf to classify it as one of the following:")
    st.write("- Blight")
    st.write("- Common Rust")
    st.write("- Gray Leaf Spot")
    st.write("- Healthy")

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.write("This app uses a pre-trained ResNet-18 model to classify plant leaf images.")
    st.sidebar.write("Make sure the image clearly shows the leaf.")

    # Device Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model
    model_path = "./model.pth"  # Replace with the actual path to your model
    model = load_model(model_path, device)

    # Define Transformations
    test_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Labels
    labels_for_viz = {0: 'Blight', 1: 'Common_Rust', 2: 'Gray_Leaf_Spot', 3: 'Healthy'}

    # Image Upload
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Classifying...")
        label = predict_image(image, model, test_transform, device, labels_for_viz)
        st.success(f"Prediction: {label}")

if __name__ == "__main__":
    main()

