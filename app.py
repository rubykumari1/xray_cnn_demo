import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# Define model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
model = SimpleCNN()
model.load_state_dict(torch.load('simple_cnn.pth', map_location=torch.device('cpu')))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Set page config FIRST
st.set_page_config(page_title="Pneumonia Classifier", page_icon="🩺")

# Custom CSS for teal-black-white theme
st.markdown("""
    <style>
    .main {
        background-color: #0A2F35; /* Dark teal */
        color: #FFFFFF; /* White text */
    }
    .stSidebar {
        background-color: #1A535C; /* Lighter teal */
    }
    .stButton>button {
        background-color: #4ECDC4; /* Teal button */
        color: #000000; /* Black text */
    }
    .prediction-box {
        background-color: #1A535C; /* Teal box */
        padding: 15px;
        border-radius: 10px;
        color: #FFFFFF;
        margin: 10px 0;
    }
    .confidence-box {
        background-color: #000000; /* Black box */
        padding: 15px;
        border-radius: 10px;
        color: #4ECDC4; /* Teal text */
        margin: 10px 0;
    }
    .model-box {
        background-color: #1A535C; /* Teal box */
        padding: 15px;
        border-radius: 10px;
        color: #FFFFFF;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app
st.title("🩺 Pneumonia X-ray Classifier")
st.markdown("Upload a 28x28 grayscale X-ray to classify it as **Pneumonia** or **Normal**. *Model*: Simple CNN trained on PneumoniaMNIST (78.37% accuracy). Built by Ruby Kumari.")

uploaded_file = st.file_uploader("Choose an X-ray...", type=["jpg", "png"], help="Upload a 28x28 grayscale image")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption='Uploaded X-ray', width=150)
    with col2:
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            prediction = "Pneumonia" if prob > 0.5 else "Normal"
            confidence = prob if prob > 0.5 else 1 - prob
        
        st.subheader("Results")
        st.markdown(f"<div class='confidence-box'><b>Confidence</b>: {confidence:.2%}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction-box'><b>Prediction</b>: {prediction}</div>", unsafe_allow_html=True)
        st.progress(int(confidence * 100))

# Sidebar with model info
with st.sidebar:
    st.header("About the Model")
    st.markdown("""
        <div class='model-box'>
            <b>Architecture</b>: 2 Conv layers (16, 32 filters), 2 FC layers (64, 1)<br>
            <b>Training</b>: 5 epochs on PneumoniaMNIST (5,856 images)<br>
            <b>Test Accuracy</b>: 78.37%<br>
            <b>Purpose</b>: Baseline for NTK generalization studies
        </div>
        """, unsafe_allow_html=True)
    st.link_button("GitHub Repo", "https://github.com/rubykumari1/xray_cnn_demo")