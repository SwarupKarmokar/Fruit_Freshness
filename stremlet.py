import streamlit as st 
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import numpy as np

from pyngrok import ngrok


class FFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(54*54*16,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,6)
        
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1,54*54*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.log_softmax(x,dim=1)

fruit_model = torch.load('FruitFreshModel.pkl')

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Fruit Freshness Classifier')
st.text('Upload the Image')

class_name = ['fresh_apple','fresh_banana','fresh_orange','rotten_apple','rotten_banana','rotten_orange']

img_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(224),
                                transforms.CenterCrop(224)])

uploaded_file = st.file_uploader("choose image: ", type=['jpeg', 'png', 'jpg', 'jfif'])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Image Uploaded Successfully')

    if st.button('PREDICT'):
        st.write('Result: ')
        new_img = img_transform(img)

        fruit_model.eval()

        with torch.no_grad():
            new_pred = fruit_model(new_img.view(1,3,224,224)).argmax()
                
        result = class_name[new_pred]
        st.title(result.upper())


