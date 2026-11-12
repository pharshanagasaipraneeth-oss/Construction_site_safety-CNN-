import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="Construction Safety Object Detection",
    page_icon="🦺",
    layout="wide",
)


# Load YOLO Model (Cached for performance)
@st.cache_resource
def load_model(model_path):
  model = YOLO(model_path)
  return model


# Update this path to point to your trained model weights (e.g., best.pt)
MODEL_PATH = "best.pt"  # Change to your actual .pt file path if needed

st.title("🦺 Construction Site Safety Object Detection")
st.write(
    "Upload an image or video to detect safety gear compliance (Hardhats,"
    " Safety Vests, Masks, Persons, etc.) using your custom-trained YOLO"
    " model."
)

# Sidebar for controls
st.sidebar.header("Configuration")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05
)

# Load model with error handling
try:
  model = load_model(MODEL_PATH)
  st.sidebar.success("Model loaded successfully!")
except Exception as e:
  st.sidebar.error(
      f"Error loading model: {e}. Please ensure '{MODEL_PATH}' is in the"
      " directory."
  )
  st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  # Open and display the original image
  image = Image.open(uploaded_file)

  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

  with col2:
    st.subheader("Detection Results")
    with st.spinner("Running inference..."):
      # Convert PIL Image to OpenCV format
      img_array = np.array(image)

      # Run YOLO inference
      results = model(img_array, conf=confidence_threshold)

      # Plot results (returns numpy array with bounding boxes drawn)
      res_plotted = results[0].plot()

      # Display processed image
      st.image(res_plotted, use_column_width=True)

  # Display detection details / class counts
  st.subheader("Detection Summary")
  boxes = results[0].boxes
  if len(boxes) > 0:
    class_names = model.names
    detected_classes = [
        class_names[int(cls)] for cls in boxes.cls.cpu().numpy()
    ]

    # Count occurrences of each detected class
    class_counts = {
        cls: detected_classes.count(cls) for cls in set(detected_classes)
    }

    # Display metrics or table
    st.write("Detected Objects Count:")
    for cls_name, count in class_counts.items():
      st.metric(label=cls_name, value=count)
  else:
    st.info("No objects detected above the confidence threshold.")