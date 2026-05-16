🧠 AI Visual Assistant

Real-time assistive computer vision prototype combining object detection, OCR, contextual hazard prioritization, object tracking, and browser-based voice feedback.

📌 Overview
AI Visual Assistant is a real-time accessibility-oriented computer vision system designed to improve environmental awareness through multimodal feedback.

The system integrates:

real-time object detection

persistent object tracking

OCR text recognition

directional awareness

contextual warning prioritization

browser-based voice alerts into a unified Streamlit interface.

Built using Python, YOLOv8, OpenCV, EasyOCR, and Streamlit.

🚀 Features

👁️ Real-Time Object Detection
Detects surrounding objects using YOLOv8 in real time.

🧭 Direction Awareness
Provides contextual direction feedback:
left
ahead
right

⚠️ Hazard Prioritization
Prioritizes nearby and important objects instead of narrating every detection.

🧠 Object Tracking
Persistent tracking IDs reduce flickering detections and improve consistency.

🗣️ Browser-Based Voice Alerts
Provides real-time spoken warnings using browser speech synthesis.

📝 OCR Text Recognition
Reads visible text from the environment using EasyOCR.

📊 FPS Monitoring
Displays real-time performance metrics.

🌙 Interactive Streamlit UI
Clean dark-themed interface designed for real-time interaction.


⚠️ Limitations

Relative distance estimation is heuristic-based.

Performance depends on CPU/GPU capability.

Browser speech synthesis behavior varies across browsers.

Designed as a prototype and research demonstration.

Not intended as a production-grade assistive navigation system.

🔮 Future Improvements

Advanced object tracking

Mobile deployment

Edge-device optimization

Smarter hazard ranking

Improved accessibility workflows

Better real-time optimization
