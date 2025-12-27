# app.py
import gradio as gr
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
    # Load model - adjust path for Hugging Face environment
    model_path = 'models/face_direction_classifier.pth'
    if not os.path.exists(model_path):
        # Try alternative paths
        model_path = '../models/face_direction_classifier.pth'
    
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
        classes = ['Front', 'Side', 'Back']
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
    
    # Create visualization
    if result.get('success', False):
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 4))
        classes = list(result['probabilities'].keys())
        probs = list(result['probabilities'].values())
        
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
        
        # Save figure to return
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        # Prepare results dictionary
        results_dict = {
            "Predicted Direction": result['predicted_class'],
            "Confidence": f"{result['confidence']:.2%}",
            "Front Probability": f"{result['probabilities'].get('Front', 0):.2%}",
            "Side Probability": f"{result['probabilities'].get('Side', 0):.2%}",
            "Back Probability": f"{result['probabilities'].get('Back', 0):.2%}"
        }
        
        return image, img_array, results_dict
    else:
        # Return error
        error_msg = result.get('error', 'Unknown error')
        return image, None, {"Error": error_msg}

def create_confidence_plot(probabilities):
    """Create a standalone plot from probabilities"""
    fig, ax = plt.subplots(figsize=(4, 3))
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    bars = ax.bar(classes, probs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Confidence Distribution')
    
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{prob:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Face Direction Classifier") as demo:
    gr.Markdown("""
    # 🧭 Face Direction Classifier
    **Upload a face image to classify its direction (Front, Side, Back)**
    
    This model uses ResNet-18 to analyze face orientation.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Face Image",
                height=300
            )
            upload_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
            clear_btn = gr.Button("🔄 Clear", variant="secondary")
            
            gr.Markdown("### 📸 Webcam Input")
            webcam = gr.Image(
                source="webcam",
                streaming=True,
                label="Live Camera Feed",
                height=200
            )
            webcam_btn = gr.Button("📷 Capture & Analyze", variant="primary")
        
        with gr.Column(scale=1):
            # Output Image with prediction overlay
            output_image = gr.Image(
                label="Uploaded Image",
                height=300,
                interactive=False
            )
            
            # Confidence plot
            confidence_plot = gr.Plot(
                label="Confidence Distribution"
            )
            
            # Results table
            results_table = gr.JSON(
                label="Prediction Results",
                value={}
            )
            
            # Quick stats
            with gr.Row():
                direction_output = gr.Label(
                    label="Predicted Direction",
                    value="",
                    show_label=True
                )
                confidence_output = gr.Label(
                    label="Confidence Score",
                    value="",
                    show_label=True
                )
    
    # Examples
    gr.Markdown("### 📁 Example Images")
    gr.Examples(
        examples=[
            ["examples/front_face.jpg"],
            ["examples/side_face.jpg"],
            ["examples/back_head.jpg"]
        ],
        inputs=image_input,
        label="Click an example to load it"
    )
    
    # Function definitions for buttons
    def analyze_image(img):
        if img is None:
            return None, None, {"Error": "Please upload an image first"}, "", ""
        
        processed_img, plot_img, results = predict_face_direction(img)
        
        if "Error" in results:
            return processed_img, None, results, "", ""
        else:
            direction = results["Predicted Direction"]
            confidence = results["Confidence"]
            return processed_img, plot_img, results, direction, confidence
    
    def clear_all():
        return None, None, {}, "", ""
    
    # Event handlers
    upload_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[output_image, confidence_plot, results_table, direction_output, confidence_output]
    )
    
    webcam_btn.click(
        fn=analyze_image,
        inputs=webcam,
        outputs=[output_image, confidence_plot, results_table, direction_output, confidence_output]
    )
    
    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[image_input, output_image, results_table, direction_output, confidence_output]
    )
    
    # Footer
    gr.Markdown("---")
    with gr.Row():
        gr.Markdown("""
        **About**: This model classifies face direction using deep learning.  
        **Model**: ResNet-18 fine-tuned on face direction dataset  
        **Classes**: Front, Side, Back
        """)
        
        gr.Markdown("""
        **Tips**:
        - Use clear, well-lit face images
        - Front-facing works best
        - Avoid extreme angles
        """)

# Launch settings for Hugging Face
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )