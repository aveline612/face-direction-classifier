# app/app.py - Production Streamlit app
import streamlit as st
from PIL import Image
import time
import logging
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import from src
import sys

sys.path.append('..')
from src.inference import get_classifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Face Direction Classifier",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/face-direction-classifier',
        'Report a bug': 'https://github.com/yourusername/face-direction-classifier/issues',
        'About': "Face Direction Classifier v1.0"
    }
)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
    st.session_state.predictions = []
    st.session_state.start_time = datetime.now()
    st.session_state.user_count = 0


def init_classifier():
    """Initialize classifier with error handling"""
    try:
        if st.session_state.classifier is None:
            with st.spinner("🚀 Loading AI model..."):
                classifier = get_classifier()
                st.session_state.classifier = classifier
                st.success("✅ Model loaded successfully!")
                logger.info("Classifier initialized")
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        logger.error(f"Classifier init failed: {str(e)}")


def log_usage(prediction_result):
    """Log prediction for analytics"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'prediction': prediction_result.get('predicted_class', 'error'),
        'confidence': prediction_result.get('confidence', 0),
        'success': prediction_result.get('success', False)
    }
    st.session_state.predictions.append(log_entry)

    # Limit history size
    if len(st.session_state.predictions) > 1000:
        st.session_state.predictions = st.session_state.predictions[-1000:]


# Initialize
init_classifier()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 5px;
        width: 100%;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🧭 Face Direction Classifier</h1>
    <p>AI-powered face direction detection (Front, Side, Back)</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)

    st.header("⚙️ Settings")

    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Minimum confidence score for reliable prediction"
    )

    # Display options
    show_details = st.checkbox("Show Detailed Results", value=True)
    show_visualization = st.checkbox("Show Visualization", value=True)
    enable_analytics = st.checkbox("Enable Usage Analytics", value=True)

    st.divider()

    # Model info
    st.header("🤖 Model Info")
    if st.session_state.classifier:
        model_info = st.session_state.classifier.get_model_info()
        st.metric("Status", "✅ Active")
        st.metric("Device", model_info['device'].upper())
        st.metric("Input Size", "224×224")
    else:
        st.warning("Model not loaded")

    st.divider()

    # Statistics
    st.header("📊 Statistics")
    st.metric("Total Predictions", len(st.session_state.predictions))
    uptime = datetime.now() - st.session_state.start_time
    st.metric("Uptime", str(uptime).split('.')[0])

    if st.button("🔄 Reload Model"):
        st.session_state.classifier = None
        init_classifier()
        st.rerun()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")

    # Upload options
    upload_option = st.radio(
        "Select input method:",
        ["Upload File", "Use Camera", "Enter URL"],
        horizontal=True
    )

    image = None
    image_source = None

    if upload_option == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a face image",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Maximum file size: 5MB"
        )
        if uploaded_file:
            # Check file size
            if uploaded_file.size > 5 * 1024 * 1024:  # 5MB
                st.error("File too large! Maximum size is 5MB.")
            else:
                image = Image.open(uploaded_file).convert('RGB')
                image_source = uploaded_file.name
                st.image(image, caption=f"📁 {uploaded_file.name}", use_container_width=True)

    elif upload_option == "Use Camera":
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            image = Image.open(camera_image).convert('RGB')
            image_source = "Camera Capture"
            st.image(image, caption="📸 Camera Capture", use_container_width=True)

    else:  # URL
        image_url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")
        if image_url:
            try:
                import requests
                from io import BytesIO

                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                    image_source = f"URL: {image_url[:50]}..."
                    st.image(image, caption="🌐 URL Image", use_container_width=True)
                else:
                    st.error("Failed to download image from URL")
            except Exception as e:
                st.error(f"Error loading URL: {str(e)}")

with col2:
    st.header("📊 Analysis Results")

    if image is not None:
        if st.button("🔍 Analyze Face Direction", type="primary", use_container_width=True):
            if st.session_state.classifier:
                # Analyze button
                with st.spinner("🤖 Analyzing face direction..."):
                    # Make prediction
                    result = st.session_state.classifier.predict(image)

                    # Log usage
                    if enable_analytics:
                        log_usage(result)

                    # Display results
                    if result['success']:
                        # Prediction card
                        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

                        # Main result
                        col_a, col_b = st.columns([2, 1])
                        with col_a:
                            direction = result['predicted_class'].upper()
                            st.markdown(f"### 🎯 **{direction}**")
                        with col_b:
                            confidence = result['confidence']
                            if confidence >= confidence_threshold:
                                st.success(f"**{confidence:.1%}**")
                            else:
                                st.warning(f"**{confidence:.1%}**")

                        # Confidence indicator
                        st.progress(
                            confidence,
                            text=f"Confidence Score: {confidence:.2%}"
                        )

                        # Performance
                        st.caption(f"⏱️ Inference time: {result['inference_time_ms']}ms")

                        st.markdown('</div>', unsafe_allow_html=True)

                        # Detailed probabilities
                        if show_details and 'probabilities' in result:
                            with st.expander("📈 Detailed Probabilities", expanded=True):
                                prob_data = result['probabilities']

                                # Create DataFrame
                                df = pd.DataFrame({
                                    'Direction': list(prob_data.keys()),
                                    'Probability': list(prob_data.values()),
                                    'Percentage': [f"{p:.2%}" for p in prob_data.values()]
                                })

                                st.dataframe(
                                    df.style.format({'Probability': '{:.4f}'}),
                                    use_container_width=True,
                                    hide_index=True
                                )

                                # Visualization
                                if show_visualization:
                                    fig = go.Figure(data=[
                                        go.Bar(
                                            x=list(prob_data.keys()),
                                            y=list(prob_data.values()),
                                            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                            text=[f"{p:.2%}" for p in prob_data.values()],
                                            textposition='auto',
                                        )
                                    ])

                                    fig.update_layout(
                                        title="Class Probabilities",
                                        xaxis_title="Face Direction",
                                        yaxis_title="Probability",
                                        yaxis_range=[0, 1],
                                        template="plotly_white"
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                        # Tips based on prediction
                        with st.expander("💡 Tips & Interpretation"):
                            if result['predicted_class'] == 'front':
                                st.info(
                                    "**Front-facing faces** are ideal for facial recognition, emotion detection, and identity verification.")
                            elif result['predicted_class'] == 'side':
                                st.info(
                                    "**Side-facing faces** are useful for pose estimation, 3D face reconstruction, and profile analysis.")
                            elif result['predicted_class'] == 'back':
                                st.info(
                                    "**Back-facing heads** can be used for head tracking, presence detection, and scene analysis.")

                            if confidence < confidence_threshold:
                                st.warning(
                                    "**Low confidence warning:** Consider uploading a clearer image with better lighting.")

                    else:
                        st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
            else:
                st.error("Model not available. Please try reloading.")

    else:
        # Empty state
        st.info("👈 **Upload an image to get started!**")

        # Demo images
        st.subheader("🎭 Try These Examples")
        demo_cols = st.columns(3)
        demo_images = {
            "Front": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop",
            "Side": "https://images.unsplash.com/photo-1519058082700-08a0b56da9b4?w-400&h=400&fit=crop",
            "Back": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&h=400&fit=crop"
        }

        for idx, (label, url) in enumerate(demo_images.items()):
            with demo_cols[idx]:
                st.image(url, caption=label, use_container_width=True)
                if st.button(f"Try {label}", key=f"demo_{idx}"):
                    st.info(f"Demo for {label} - upload your own image to test!")

# Analytics Dashboard
if enable_analytics and st.session_state.predictions:
    st.divider()
    st.header("📈 Usage Analytics")

    if len(st.session_state.predictions) > 1:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.predictions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(df))
        with col2:
            success_rate = df['success'].mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            avg_confidence = df[df['success']]['confidence'].mean() * 100
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        with col4:
            popular_class = df[df['success']]['prediction'].mode()[0] if not df[df['success']].empty else "N/A"
            st.metric("Most Common", popular_class.upper())

        # Distribution chart
        if not df[df['success']].empty:
            fig = px.pie(
                df[df['success']],
                names='prediction',
                title='Prediction Distribution',
                color='prediction',
                color_discrete_map={'front': '#4ECDC4', 'side': '#45B7D1', 'back': '#FF6B6B'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Show recent predictions
        with st.expander("📋 View Recent Predictions"):
            st.dataframe(df.tail(10), use_container_width=True)

# Footer
st.divider()
st.markdown(f"""
<div class="footer">
    <p><strong>Face Direction Classifier v1.0</strong> | Built with PyTorch & Streamlit</p>
    <p>© {datetime.now().year} | Model: ResNet-18 | Input: 224×224 RGB</p>
    <p>For support, visit <a href="https://github.com/yourusername/face-direction-classifier">GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)