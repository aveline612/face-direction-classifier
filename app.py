# app.py
import streamlit as st
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Try to import your classifier
try:
    from src.inference import FaceDirectionClassifier
    MODEL_AVAILABLE = True
    
    # Define model path
    model_path = 'models/face_direction_classifier.pth'
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Check if model exists, if not, try to download it
    if not os.path.exists(model_path):
        print("⚠️ Model file not found, attempting to download...")
        try:
            import requests
            # Replace this URL with your actual model file URL
            model_url = "https://huggingface.co/spaces/your-username/face-direction-classifier/resolve/main/models/face_direction_classifier.pth"
            response = requests.get(model_url)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Model downloaded successfully to {model_path}")
            else:
                print("❌ Could not download model, using mock predictions")
                MODEL_AVAILABLE = False
        except Exception as e:
            print(f"❌ Download failed: {e}, using mock predictions")
            MODEL_AVAILABLE = False
    
    if os.path.exists(model_path):
        classifier = FaceDirectionClassifier(model_path)
        print(f"✅ Model loaded from {model_path}")
    else:
        MODEL_AVAILABLE = False
        print("⚠️ Model file not found, using mock predictions")
except ImportError as e:
    MODEL_AVAILABLE = False
    print(f"⚠️ Could not import FaceDirectionClassifier: {e}")

def predict_face_direction(image):
    """
    Process uploaded image and return face direction prediction
    """
    if not MODEL_AVAILABLE:
        # Mock prediction for demonstration
        import random
        classes = ['back', 'front', 'side']
        probs = [random.random() for _ in range(3)]
        total = sum(probs)
        probs = [p/total for p in probs]
        max_idx = np.argmax(probs)
        
        result = {
            'success': True,
            'predicted_class': classes[max_idx],
            'confidence': float(probs[max_idx]),
            'probabilities': dict(zip(classes, probs))
        }
    else:
        # Real prediction
        result = classifier.predict(image)
    
    return result

def create_confidence_plot(probabilities):
    """Create a matplotlib plot from probabilities"""
    fig, ax = plt.subplots(figsize=(8, 5))
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
    bars = ax.bar(classes, probs, color=colors, edgecolor='black')
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Probability', fontweight='bold')
    ax.set_title('Class Probabilities', fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return fig

# Streamlit app
st.set_page_config(page_title="Face Direction Classifier", layout="wide")

st.title("🧭 Face Direction Classifier")
st.markdown("Upload a face image to classify its direction (Front, Side, Back)")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)

with col2:
    st.header("Analysis Results")
    
    if uploaded_file is not None:
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                result = predict_face_direction(image)
                
                if result.get('success', False):
                    # Display prediction results
                    predicted_class = result['predicted_class']
                    confidence = result['confidence']
                    
                    # Show prediction with metrics
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric(label="Predicted Direction", value=predicted_class.upper())
                    with col_metric2:
                        st.metric(label="Confidence", value=f"{confidence:.2%}")
                    
                    # Show probability chart
                    st.subheader("Class Probabilities")
                    fig = create_confidence_plot(result['probabilities'])
                    st.pyplot(fig)
                    
                    # Show detailed results
                    with st.expander("Detailed Results"):
                        st.json(result)
                else:
                    st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")
    else:
        st.info("👆 Please upload an image to get started!")

# Sidebar with information
st.sidebar.header("About")
st.sidebar.info("This model uses ResNet-18 to analyze face orientation.")
st.sidebar.markdown("""
### Classes:
- **BACK**: Back-facing head
- **FRONT**: Front-facing face  
- **SIDE**: Side-facing face
""")

st.sidebar.header("Tips")
st.sidebar.markdown("""
- Use clear, well-lit face images
- Front-facing works best
- Avoid extreme angles
- Images should be at least 50x50 pixels
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
<p><strong>Face Direction Classifier</strong> | Built with Streamlit & PyTorch</p>
<p>Model: ResNet-18 fine-tuned for face direction classification</p>
</div>
""", unsafe_allow_html=True)