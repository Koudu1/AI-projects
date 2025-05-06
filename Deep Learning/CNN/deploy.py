import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class leNetClassifer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ConvLayer1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding='same')
        self.PoolLayer = torch.nn.AvgPool2d(kernel_size=2)
        self.ConvLayer2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.ConvLayer3 = torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.Linear1 = torch.nn.Linear(in_features=120, out_features=84)
        self.Linear2 = torch.nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        x = torch.nn.functional.tanh(self.ConvLayer1(x))
        x = self.PoolLayer(x)
        x = torch.nn.functional.tanh(self.ConvLayer2(x))
        x = self.PoolLayer(x)
        x = torch.nn.functional.tanh(self.ConvLayer3(x))
        x = torch.nn.Flatten()(x)
        x = self.Linear1(x)
        x = torch.nn.functional.tanh(x)
        outputs = self.Linear2(x)

        return outputs
    
@st.cache_resource
def load_model(model_path, num_classes = 10):
    leNet = leNetClassifer(num_classes)
    leNet.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    leNet.eval()
    return leNet

model = load_model('D:\Self learn programming\AI projects\Deep Learning\CNN\model\leNet.pt')

def inference(image, model):
    w, h = image.size
    if w != h:
        crop = transforms.CenterCrop(min(w, h))
        image = crop(image)
        wnew, hnew = image.size
    
    img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])

    img_new = img_transform(image)
    img_new = img_new.expand(1, 1, 28, 28)
    with torch.no_grad():
        outputs = model(img_new)

    preds = nn.Softmax(dim=1)(outputs)
    p_max, yhat = torch.max(preds.data, 1)
    return p_max.item()*100 , yhat.item() 

def main():
    st.title('Digit Recognition')
    st.subheader('Model: LeNet. Dataset: MNIST')
    option = st.selectbox('How would you like to give the input?', ('Upload Image File', 'Run Example Image'))
    if option == 'Upload Image File':
        file = st.file_uploader("Please upload your file", type=['jpg', 'png'])
        if file is not None:
            image = Image.open(file)
            p, label = inference(image, model)
            st.image(image)
            st.success(f"The upload image is of digit {label} with {p:.2f}% probability")
    
    elif option == 'Run Example Image':
        image = Image.open('D:\Self learn programming\AI projects\Deep Learning\CNN\demo_8.png')
        p, label = inference(image, model)
        st.image(image)
        st.success(f"The upload image is of digit {label} with {p:.2f}% probability")

if __name__ == '__main__':
    main()