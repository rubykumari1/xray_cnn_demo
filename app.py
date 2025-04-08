import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# Define model (must match your Colab CNN)
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

# Preprocessing (for 28x28 PneumoniaMNIST images)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Streamlit app
st.title("Pneumonia X-ray Classifier")
st.write("Upload a 28x28 grayscale X-ray to classify (e.g., from PneumoniaMNIST).")

uploaded_file = st.file_uploader("Choose an X-ray...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Grayscale
    st.image(image, caption='Uploaded X-ray', use_column_width=True)
    
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        prediction = "Pneumonia" if prob > 0.5 else "Normal"
        confidence = prob if prob > 0.5 else 1 - prob
    
    st.write(f"**Prediction**: {prediction}")
    st.write(f"**Confidence**: {confidence:.2%}")