import streamlit as st
from PIL import Image
from torchvision import transforms,models
import torch.nn as nn
import torch
st.title("BRAIN TUMOUR CLASSIFIER")
st.write("Upload image of brain MRI scan")
file=st.file_uploader("choose a jpg image",type=["jpg"] )
transform_jpg=transforms.Compose([transforms.Resize((244,244)),transforms.ToTensor()])
resnet=models.resnet50(pretrained=True)
resnet.fc=nn.Sequential(
    nn.Linear(2048,512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.4),
    nn.Linear(512,4)
)
idx_class={0: 'glioma', 1: 'meningioma', 2: 'no_tumor', 3: 'pituitary'}
state_dict=torch.load("./resnet-92.pth")
resnet.load_state_dict(state_dict=state_dict)
resnet.eval()
if file is not None:
    image=Image.open(file)
    image1=transform_jpg(image)
    image1=image1.unsqueeze(0)
    out=resnet(image1)
    i=torch.argmax(out).item()
    st.image(image=image,caption=f"the class predicted:{idx_class[i]}")