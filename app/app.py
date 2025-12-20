# app.py
import streamlit as st
from face_direction_inference import FaceDirectionClassifier
from PIL import Image
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Face Direction Classifier", layout="wide")


@st.cache_resource
def load_model():
    return FaceDirectionClassifier('face_direction_classifier.pth')


classifier = load_model()

st.title("🧭 Face Direction Classifier")
st.markdown("Upload a face image to classify direction (Front, Side, Back)")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    if uploaded_file:
        if st.button("🔍 Analyze", type="primary"):
            with st.spinner("Analyzing..."):
                result = classifier.predict(image)

                if result['success']:
                    st.success(f"✅ Prediction: **{result['predicted_class'].upper()}**")
                    st.metric("Confidence", f"{result['confidence']:.2%}")

                    # Show probabilities
                    fig, ax = plt.subplots()
                    classes = list(result['probabilities'].keys())
                    probs = list(result['probabilities'].values())
                    bars = ax.bar(classes, probs, color=['red', 'green', 'blue'])
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Probability')
                    st.pyplot(fig)

                    # Show details
                    with st.expander("Detailed Results"):
                        st.json(result)
                else:
                    st.error(f"Prediction failed: {result.get('error')}")

st.sidebar.markdown("### About")
st.sidebar.info("This model classifies face direction using ResNet-18")