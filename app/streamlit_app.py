import streamlit as st
import cv2
from ultralytics import YOLO
import streamlit.components.v1 as components
import easyocr
import time

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="AI Visual Assistant",
    page_icon="🧠",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================
st.markdown(
    """
    <style>

    .main {
        background-color: #050816;
        color: white;
    }

    h1 {
        color: white !important;
        font-size: 52px !important;
        font-weight: 800 !important;
    }

    .stButton > button {
        background-color: #6C63FF;
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        border: none;
        font-size: 16px;
        font-weight: 600;
    }

    .metric-box {
        background-color: #111827;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================
# LOAD MODELS
# =========================================
model = YOLO("yolov8n.pt")

reader = easyocr.Reader(['en'])

# =========================================
# SESSION STATE
# =========================================
if "last_spoken" not in st.session_state:
    st.session_state.last_spoken = ""

if "last_spoken_time" not in st.session_state:
    st.session_state.last_spoken_time = 0

# =========================================
# HEADER
# =========================================
st.title("🧠 AI Visual Assistant")

st.markdown(
    "Real-time assistive computer vision system"
)

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:

    st.header("⚙️ Controls")

    run = st.checkbox("Start Camera")

    read_text_button = st.button(
        "📖 Read Visible Text"
    )

    st.markdown("---")

    st.subheader("📌 Features")

    st.write("✅ Object Detection")
    st.write("✅ Object Tracking")
    st.write("✅ Voice Alerts")
    st.write("✅ OCR")
    st.write("✅ Prioritized Warnings")
    st.write("✅ Direction Awareness")
    st.write("✅ FPS Monitoring")

# =========================================
# MAIN LAYOUT
# =========================================
left_col, right_col = st.columns([3, 1])

with left_col:

    frame_placeholder = st.empty()

with right_col:

    st.markdown("### 🚨 Live Detection")
    warning_placeholder = st.empty()

    st.markdown("### 📝 OCR Output")
    ocr_placeholder = st.empty()

    st.markdown("### 📊 System Stats")

    object_count_placeholder = st.empty()

    fps_placeholder = st.empty()

# =========================================
# CAMERA SETUP
# =========================================
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================================
# PERFORMANCE VARIABLES
# =========================================
prev_time = time.time()

frame_count = 0

# =========================================
# MAIN LOOP
# =========================================
while run:

    ret, frame = camera.read()

    if not ret:
        st.error("Failed to access webcam")
        break

    # =====================================
    # FRAME SKIPPING
    # =====================================
    frame_count += 1

    if frame_count % 3 != 0:
        continue

    # =====================================
    # RESIZE FRAME
    # =====================================
    frame = cv2.resize(frame, (416, 320))

    # =====================================
    # YOLO TRACKING
    # =====================================
    results = model.track(
        frame,
        persist=True,
        imgsz=224,
        verbose=False
    )

    boxes = results[0].boxes

    priority_objects = []

    # =====================================
    # PROCESS DETECTIONS
    # =====================================
    for box in boxes:

        cls = int(box.cls[0])

        label = model.names[cls]

        # -----------------------------
        # CONFIDENCE FILTER
        # -----------------------------
        confidence = float(box.conf[0])

        if confidence < 0.5:
            continue

        # -----------------------------
        # TRACKING ID
        # -----------------------------
        track_id = None

        if box.id is not None:
            track_id = int(box.id[0])

        # -----------------------------
        # BOUNDING BOX
        # -----------------------------
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        width = x2 - x1
        height = y2 - y1

        area = width * height

        # -----------------------------
        # DISTANCE ESTIMATION
        # -----------------------------
        if area > 50000:

            danger = "🔴 WARNING"
            distance = "very close"

        elif area > 20000:

            danger = "🟡 Nearby"
            distance = "nearby"

        else:

            danger = "🟢 Safe"
            distance = "far"

        # -----------------------------
        # DIRECTION ESTIMATION
        # -----------------------------
        object_center = (x1 + x2) // 2

        frame_width = frame.shape[1]

        if object_center < frame_width // 3:

            direction = "left"

        elif object_center < 2 * frame_width // 3:

            direction = "ahead"

        else:

            direction = "right"

        # -----------------------------
        # PRIORITY SCORE
        # -----------------------------
        priority_score = 0

        # Humans prioritized
        if label == "person":
            priority_score += 100

        # Larger objects prioritized
        priority_score += area // 1000

        # Central objects prioritized
        frame_center = frame_width // 2

        center_distance = abs(
            object_center - frame_center
        )

        priority_score += max(
            0,
            100 - center_distance
        )

        # -----------------------------
        # DISPLAY LABEL
        # -----------------------------
        final_label = (
            f"{danger}: {label}"
            f" #{track_id} "
            f"{distance} {direction}"
        )

        # -----------------------------
        # STORE OBJECT
        # -----------------------------
        priority_objects.append({

            "text": final_label,

            "score": priority_score,

            "danger": danger,

            "label": label,

            "short_speech":
                f"{label} {direction}"
        })

    # =====================================
    # SORT OBJECTS BY PRIORITY
    # =====================================
    priority_objects = sorted(

        priority_objects,

        key=lambda x: x["score"],

        reverse=True
    )

    # =====================================
    # DISPLAY LIST
    # =====================================
    detected_objects = [

        obj["text"]

        for obj in priority_objects
    ]

    unique_objects = list(
        dict.fromkeys(detected_objects)
    )

    # =====================================
    # WARNING PANEL
    # =====================================
    if unique_objects:

        warning_text = (
            "Detected:\n\n"
            + "\n".join(unique_objects)
        )

        warning_placeholder.warning(
            warning_text
        )

        object_count_placeholder.markdown(
            f"""
            <div class="metric-box">
                <h2>{len(unique_objects)}</h2>
                <p>Objects Detected</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # =================================
        # SPEAK TOP WARNING ONLY
        # =================================
        spoken_text = ""

        top_object = priority_objects[0]

        if top_object["danger"] == "🔴 WARNING":

            spoken_text = (
                "Warning "
                + top_object["short_speech"]
            )

        current_time = time.time()

        # =================================
        # SPEECH COOLDOWN
        # =================================
        if (
            spoken_text
            and spoken_text != st.session_state.last_spoken
            and current_time
            - st.session_state.last_spoken_time > 3
        ):

            components.html(
                f"""
                <script>

                window.speechSynthesis.cancel();

                var msg =
                    new SpeechSynthesisUtterance(
                        "{spoken_text}"
                    );

                window.speechSynthesis.speak(msg);

                </script>
                """,
                height=0,
            )

            st.session_state.last_spoken = (
                spoken_text
            )

            st.session_state.last_spoken_time = (
                current_time
            )

    else:

        warning_placeholder.info(
            "No objects detected"
        )

    # =====================================
    # OCR BUTTON
    # =====================================
    if read_text_button:

        results_ocr = reader.readtext(frame)

        extracted_texts = []

        for result in results_ocr:

            extracted_texts.append(
                result[1]
            )

        if extracted_texts:

            final_text = " ".join(
                extracted_texts
            )

            ocr_placeholder.success(
                "Detected Text:\n\n"
                + final_text
            )

            components.html(
                f"""
                <script>

                window.speechSynthesis.cancel();

                var msg =
                    new SpeechSynthesisUtterance(
                        "{final_text}"
                    );

                window.speechSynthesis.speak(msg);

                </script>
                """,
                height=0,
            )

        else:

            ocr_placeholder.warning(
                "No text detected"
            )

    # =====================================
    # DRAW RESULTS
    # =====================================
    annotated_frame = results[0].plot()

    # =====================================
    # FPS CALCULATION
    # =====================================
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # =====================================
    # DRAW FPS
    # =====================================
    cv2.putText(

        annotated_frame,

        f"FPS: {int(fps)}",

        (10, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 0),

        2
    )

    fps_placeholder.markdown(

        f"""
        <div class="metric-box">
            <h2>{int(fps)}</h2>
            <p>FPS</p>
        </div>
        """,

        unsafe_allow_html=True
    )

    # =====================================
    # DISPLAY FRAME
    # =====================================
    annotated_frame = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB
    )

    frame_placeholder.image(
        annotated_frame,
        channels="RGB"
    )

# =========================================
# CLEANUP
# =========================================
camera.release()


