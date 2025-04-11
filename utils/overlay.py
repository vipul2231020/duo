import cv2
import mediapipe as mp
import numpy as np

# Load face detector
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Load model image
model_img = cv2.imread("model.jpg")
model_img_rgb = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)

# Load glasses image (with transparent background if possible)
glasses_img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)  # make sure this is RGBA

# Detect facial landmarks
results = face_mesh.process(model_img_rgb)
if not results.multi_face_landmarks:
    print("No face found!")
    exit()

# Get landmarks for eyes (left & right)
landmarks = results.multi_face_landmarks[0].landmark
img_h, img_w, _ = model_img.shape

left_eye = [int(landmarks[33].x * img_w), int(landmarks[33].y * img_h)]
right_eye = [int(landmarks[263].x * img_w), int(landmarks[263].y * img_h)]

# Calculate placement position
eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
glasses_width = int(np.linalg.norm(np.array(left_eye) - np.array(right_eye)) * 2)

# Resize glasses
scale = glasses_width / glasses_img.shape[1]
new_h = int(glasses_img.shape[0] * scale)
resized_glasses = cv2.resize(glasses_img, (glasses_width, new_h), interpolation=cv2.INTER_AREA)

# Overlay function
def overlay_transparent(background, overlay, x, y):
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]

    if x >= bw or y >= bh:
        return background

    if x + w > bw:
        w = bw - x
        overlay = overlay[:, :w]
    if y + h > bh:
        h = bh - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate([overlay, np.ones((h, w, 1), dtype=overlay.dtype) * 255], axis=2)

    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_img
    return background.astype(np.uint8)

# Place glasses above eye center
x = eye_center[0] - resized_glasses.shape[1] // 2
y = eye_center[1] - resized_glasses.shape[0] // 2
output = overlay_transparent(model_img.copy(), resized_glasses, x, y)

# Save & Show
cv2.imwrite("output_auto.png", output)
cv2.imshow("Auto Glasses Placement", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
