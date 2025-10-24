# ai_vision_live.py
import time
import cv2
import numpy as np

# --- Flags de módulos (podem ser alternadas ao vivo com o teclado) ---
USE_OBJECTS = True
USE_FACES   = True
USE_EMOTION = True
USE_HANDS   = True

# --- YOLO (Ultralytics) - Objetos ---
yolo_model = None
yolo_names = None
try:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")  # baixa na 1ª execução
    # classes do modelo
    yolo_names = yolo_model.names
except Exception as e:
    print("[AVISO] YOLO não disponível:", e)

# --- MediaPipe Face Detection ---
mp_face = None
face_det = None
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    face_det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
except Exception as e:
    print("[AVISO] MediaPipe FaceDetection não disponível:", e)

# --- MediaPipe Hands (gestos) ---
mp_hands = None
hands = None
mp_draw = None
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
except Exception as e:
    print("[AVISO] MediaPipe Hands não disponível:", e)

# --- DeepFace (emoções) ---
DeepFace = None
try:
    from deepface import DeepFace
except Exception as e:
    print("[AVISO] DeepFace não disponível:", e)

# ---------- Utilidades de desenho ----------
def draw_fancy_box(img, x1, y1, x2, y2, color=(0, 255, 0), thickness=2, r=8, d=16):
    """Borda estilizada (cantos) para destacar box."""
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    # cantos
    cv2.line(img, (x1, y1), (x1 + d, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + d), color, thickness)
    cv2.line(img, (x2, y1), (x2 - d, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + d), color, thickness)
    cv2.line(img, (x1, y2), (x1 + d, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - d), color, thickness)
    cv2.line(img, (x2, y2), (x2 - d, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - d), color, thickness)

def put_tag(img, text, org, bg=(0,0,0), fg=(255,255,255)):
    x, y = org
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y + 6), bg, -1)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 2, cv2.LINE_AA)

# ---------- Emoções (amostragem a cada N frames p/ performance) ----------
EMO_EVERY_N = 7
last_emo = ""
def analyze_emotion(face_bgr):
    global last_emo
    if DeepFace is None:
        return ""
    try:
        # DeepFace espera RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        res = DeepFace.analyze(face_rgb, actions=["emotion"], enforce_detection=False, prog_bar=False)
        # DeepFace pode retornar dict ou list dependendo da versão
        if isinstance(res, list) and len(res) > 0:
            res = res[0]
        emo = res.get("dominant_emotion", "")
        last_emo = emo
        return emo
    except Exception:
        return last_emo

# ---------- Gestos simples (contagem de dedos) ----------
def fingers_state(landmarks, w, h, handedness="Right"):
    """
    Retorna lista [thumb, index, middle, ring, pinky] (1 levantado, 0 abaixado).
    Regra simples; para polegar consideramos direção por X e lado da mão.
    """
    lm = landmarks
    dedos = []

    # Polegar
    thumb_tip_x = lm[4].x * w
    thumb_ip_x  = lm[3].x * w
    if handedness == "Right":
        dedos.append(1 if thumb_tip_x > thumb_ip_x else 0)
    else:  # Left (espelho)
        dedos.append(1 if thumb_tip_x < thumb_ip_x else 0)

    # Indicador, Médio, Anelar, Mindinho (ponta acima da junta)
    tips = [8, 12, 16, 20]
    pins = [6, 10, 14, 18]
    for t, p in zip(tips, pins):
        dedos.append(1 if lm[t].y < lm[p].y else 0)

    return dedos

def gesture_name(dedos):
    s = sum(dedos)
    if dedos == [1,0,0,0,0]: return "JOINHA"
    if dedos == [0,1,0,0,0]: return "INDICADOR"
    if dedos == [0,1,1,0,0]: return "V (PAZ)"
    if s == 5: return "MÃO ABERTA"
    if s == 0: return "PUNHO"
    return f"{s} dedos"

# ---------- Loop da câmera ----------
# Use a video file instead of the camera in Colab
cap = cv2.VideoCapture('/content/sample_data/cat.mp4') # Replace with your video file path
if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir a câmera ou arquivo de vídeo.")

# Otimizações - These might not be applicable to a video file
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev = time.time()
frame_id = 0

while True:
    ok, frame = cap.read()
    if not ok: break
    frame_id += 1
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # FPS
    now = time.time()
    fps = 1.0 / (now - prev) if now != prev else 0.0
    prev = now

    # ---------- OBJETOS ----------
    if USE_OBJECTS and yolo_model is not None:
        try:
            # Resize frame for faster processing with YOLO if needed
            # frame_resized = cv2.resize(frame, (640, 640))
            results = yolo_model.predict(frame, conf=0.5, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    name = yolo_names.get(cls, str(cls)) if yolo_names else str(cls)
                    draw_fancy_box(frame, x1, y1, x2, y2, color=(0, 200, 255), thickness=2)
                    put_tag(frame, f"{name} {conf:.2f}", (int(x1), int(y1)-6), bg=(0, 0, 0), fg=(255, 255, 255))
        except Exception as e:
            put_tag(frame, f"YOLO erro: {str(e)[:28]}", (10, 60), bg=(0,0,255))


    # ---------- FACES + EMOÇÃO ----------
    faces_xyxy = []
    if USE_FACES and face_det is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = face_det.process(rgb)
        if out and out.detections:
            for det in out.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                # clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                faces_xyxy.append((x1, y1, x2, y2))
                draw_fancy_box(frame, x1, y1, x2, y2, color=(0,255,0), thickness=2)
                put_tag(frame, "Face", (x1, y1 - 6), bg=(0, 128, 0))

                # Emoções (a cada EMO_EVERY_N frames)
                if USE_EMOTION and ((frame_id % EMO_EVERY_N) == 0):
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        emo = analyze_emotion(face_crop)
                        if emo:
                            put_tag(frame, f"Emoção: {emo}", (x1, y2 + 24), bg=(60,60,60))


    # ---------- MÃOS / GESTOS ----------
    if USE_HANDS and hands is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            # handedness pode estar em res.multi_handedness
            handed_info = []
            try:
                handed_info = [h.classification[0].label for h in res.multi_handedness]
            except Exception:
                handed_info = ["Right"] * len(res.multi_hand_landmarks)

            for idx, hand_lm in enumerate(res.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                dedos = fingers_state(hand_lm.landmark, w, h, handed_info[idx] if idx < len(handed_info) else "Right")
                name = gesture_name(dedos)
                # Pegar a ponta do indicador para posicionar a tag
                px = int(hand_lm.landmark[8].x * w)
                py = int(hand_lm.landmark[8].y * h)
                put_tag(frame, f"{name}", (max(0, px-50), max(0, py-10)), bg=(40,40,40))


    # ---------- HUD ----------
    put_tag(frame, f"FPS: {fps:.1f}", (10, 30), bg=(0,0,0))
    status = f"[O:{'ON' if USE_OBJECTS else 'off'} F:{'ON' if USE_FACES else 'off'} E:{'ON' if USE_EMOTION else 'off'} H:{'ON' if USE_HANDS else 'off'}]"
    put_tag(frame, status, (10, h-10), bg=(0,0,0))

    # In Colab, you can display the image using imshow, but it won't show a live window.
    # You might need to use cv2_imshow from google.colab.patches for image display.
    # However, for a video loop, displaying each frame might be slow.
    # You can uncomment the line below if you install google.colab.patches and import cv2_imshow
    # from google.colab.patches import cv2_imshow
    # cv2_imshow(frame)

    # waitKey is still needed to control the loop and handle key presses
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('o'):
        USE_OBJECTS = not USE_OBJECTS
    elif k == ord('f'):
        USE_FACES = not USE_FACES
    elif k == ord('e'):
        USE_EMOTION = not USE_EMOTION
    elif k == ord('h'):
        USE_HANDS = not USE_HANDS


cap.release()
cv2.destroyAllWindows()