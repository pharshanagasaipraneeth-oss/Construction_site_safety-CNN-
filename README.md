# 🦺 Construction Site Safety Object Detection

A real-time web application built with **Streamlit** and **Ultralytics YOLOv8** to detect construction site safety compliance (Hardhats, Safety Vests, Masks, Persons, etc.) from uploaded images.

---

## 🚀 Features
- **Interactive Web Interface:** Clean and responsive UI powered by Streamlit.
- **Adjustable Confidence Threshold:** Dynamically filter detection sensitivity via the sidebar slider.
- **Side-by-Side Comparison:** View the original image next to the annotated detection results.
- **Detection Summary:** Automatically counts and displays detected safety gear and objects.

---

## 🛠️ Tech Stack
- **Python 3.8+**
- **Streamlit** (Frontend Web App)
- **Ultralytics YOLO** (Object Detection Model)
- **OpenCV & Pillow** (Image Processing)
- **NumPy** (Numerical Operations)

---

## 📂 Project Structure
```text
├── app.py              # Main Streamlit application script
├── best.pt             # Custom-trained YOLO model weights
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
