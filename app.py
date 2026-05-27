import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import squeezenet1_0
from fastai.vision.learner import create_vision_model

st.title("OHT Composite Failure Classifier")

@st.cache_resource
def load_model():
    ckpt = torch.load("oht_model_2.pth", map_location="cpu")
    vocab = ckpt["vocab"]

    model = create_vision_model(
        squeezenet1_0,
        n_out=len(vocab),
        pretrained=False
    )

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab

model, vocab = load_model()

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(img):
    x = tfm(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    pred = vocab[pred_idx]
    conf = probs[pred_idx].item()

    st.image(img, caption="Selected image", use_container_width=True)
    st.subheader(f"Prediction: {pred.upper()}")
    st.write(f"Confidence: {conf:.2%}")

st.markdown("### Try sample images")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Safe sample"):
        img = Image.open("samples/safe.png").convert("RGB")
        predict_image(img)

with col2:
    if st.button("Moderate sample"):
        img = Image.open("samples/moderate.png").convert("RGB")
        predict_image(img)

with col3:
    if st.button("Failed sample"):
        img = Image.open("samples/failed.png").convert("RGB")
        predict_image(img)

st.markdown("---")
st.markdown("### Or upload your own image")

uploaded_file = st.file_uploader("Upload OHT contour image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    predict_image(img)
