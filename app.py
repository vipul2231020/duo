import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

st.title("🕶️ Virtual Glasses Try-On")

uploaded_model = st.file_uploader("Upload Model Image", type=["jpg", "png"])
uploaded_glasses = st.file_uploader("Upload Glasses (PNG with Transparent BG)", type=["png"])

if uploaded_model and uploaded_glasses:
    model = Image.open(uploaded_model).convert("RGB")
    model_cv = np.array(model)
    glasses = Image.open(uploaded_glasses).convert("RGBA")
    glasses_cv = np.array(glasses)

    # Face Detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    results = face_mesh.process(cv2.cvtColor(model_cv, cv2.COLOR_RGB2BGR))

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        h, w, _ = model_cv.shape
        left = face_landmarks.landmark[33]
        right = face_landmarks.landmark[263]

        x1, y1 = int(left.x * w), int(left.y * h)
        x2, y2 = int(right.x * w), int(right.y * h)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = int(1.5 * np.linalg.norm([x2 - x1, y2 - y1]))
        scale = width / glasses_cv.shape[1]
        new_h = int(glasses_cv.shape[0] * scale)

        # Resize glasses
        resized_glasses = cv2.resize(glasses_cv, (width, new_h), interpolation=cv2.INTER_AREA)

        # Overlay
        top_left_x = int(center_x - width // 2)
        top_left_y = int(center_y - new_h // 2)

        for i in range(resized_glasses.shape[0]):
            for j in range(resized_glasses.shape[1]):
                if top_left_y + i >= model_cv.shape[0] or top_left_x + j >= model_cv.shape[1] or top_left_x + j < 0:
                    continue
                alpha = resized_glasses[i, j, 3] / 255.0
                if alpha > 0:
                    model_cv[top_left_y + i, top_left_x + j] = (
                        alpha * resized_glasses[i, j, :3] +
                        (1 - alpha) * model_cv[top_left_y + i, top_left_x + j]
                    )

        st.image(model_cv.astype(np.uint8), caption="With Glasses", use_column_width=True)
    else:
        st.warning("Face not detected. Try a front-facing image.")
