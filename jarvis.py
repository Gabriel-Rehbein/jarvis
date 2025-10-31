# ai_vision_live.py  — versão aprimorada e otimizada (Windows-friendly)
import sys, os, time
import cv2
import numpy as np

# --- FLAGS INICIAIS ---
USE_OBJECTS = True
USE_FACES = True
USE_EMOTION = True
USE_HANDS = True

# ==============================
#  ABERTURA DE FONTE DE VÍDEO
# ==============================
def open_source(video_path=None):
    """Abre arquivo ou câmera com fallback em DirectShow e MediaFoundation."""
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            print(f"[INFO] Arquivo de vídeo aberto: {video_path}")
            return cap, f"FILE:{video_path}"

    for api, name in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]:
        for i in range(6):
            cap = cv2.VideoCapture(i, api)
            if cap.isOpened():
                print(f"[INFO] Câmera detectada via {name}:{i}")
                return cap, f"{name}:{i}"
            cap.release()

    return None, None


# ==============================
#  CARREGAMENTO DE MODELOS
# ==============================
def load_models():
    """Carrega YOLO, MediaPipe e DeepFace com segurança."""
    global yolo_model, yolo_names, face_det, hands, mp_draw, DeepFace

    # YOLO
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")
        yolo_names = yolo_model.names
        print("[INFO] YOLO carregado.")
    except Exception as e:
        print("[AVISO] YOLO não disponível:", e)
        yolo_model = yolo_names = None

    # MediaPipe Face
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        face_det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        print("[INFO] MediaPipe Face Detection carregado.")
    except Exception as e:
        print("[AVISO] MediaPipe FaceDetection não disponível:", e)
        face_det = None

    # MediaPipe Hands
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(False, 2, 0.7, 0.5)
        mp_draw = mp.solutions.drawing_utils
        print("[INFO] MediaPipe Hands carregado.")
    except Exception as e:
        print("[AVISO] MediaPipe Hands não disponível:", e)
        hands = mp_draw = None

    # DeepFace
    try:
        from deepface import DeepFace
        print("[INFO] DeepFace carregado.")
    except Exception as e:
        print("[AVISO] DeepFace não disponível:", e)
        DeepFace = None


# ==============================
#  FUNÇÕES DE DESENHO
# ==============================
def draw_fancy_box(img, x1, y1, x2, y2, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    for (px, py) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.circle(img, (px, py), 4, color, -1)


def put_tag(img, text, pos, bg=(0, 0, 0), fg=(255, 255, 255)):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y + 6), bg, -1)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 2, cv2.LINE_AA)


# ==============================
#  ANÁLISE DE EMOÇÕES
# ==============================
last_emo = ""
def analyze_emotion(face_bgr):
    global last_emo
    if DeepFace is None:
        return ""
    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        res = DeepFace.analyze(face_rgb, actions=["emotion"], enforce_detection=False, prog_bar=False)
        if isinstance(res, list) and res:
            res = res[0]
        last_emo = res.get("dominant_emotion", last_emo)
        return last_emo
    except Exception:
        return last_emo


# ==============================
#  GESTOS DAS MÃOS
# ==============================
def fingers_state(lm, w, h, handed="Right"):
    thumb_tip_x, thumb_ip_x = lm[4].x * w, lm[3].x * w
    dedos = [1 if (thumb_tip_x > thumb_ip_x if handed == "Right" else thumb_tip_x < thumb_ip_x) else 0]
    for t, p in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        dedos.append(1 if lm[t].y < lm[p].y else 0)
    return dedos


def gesture_name(dedos):
    s = sum(dedos)
    if dedos == [1, 0, 0, 0, 0]: return "JOINHA 👍"
    if dedos == [0, 1, 1, 0, 0]: return "PAZ ✌️"
    if s == 0: return "PUNHO ✊"
    if s == 5: return "ABERTA ✋"
    return f"{s} dedos"


# ==============================
#  LOOP PRINCIPAL
# ==============================
def main():
    global USE_OBJECTS, USE_FACES, USE_EMOTION, USE_HANDS
    video_arg = sys.argv[sys.argv.index("--video")+1] if "--video" in sys.argv else None
    cap, src = open_source(video_arg)
    if not cap:
        raise RuntimeError("Não foi possível abrir câmera ou vídeo.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("AI Vision", cv2.WINDOW_NORMAL)
    prev_time, frame_id = time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[AVISO] Sem sinal de vídeo.")
            break
        frame_id += 1
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # FPS
        now = time.time()
        fps = 1.0 / max((now - prev_time), 1e-5)
        prev_time = now

        # ====== DETECÇÃO DE OBJETOS ======
        if USE_OBJECTS and yolo_model:
            try:
                for r in yolo_model.predict(frame, conf=0.5, verbose=False):
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls = int(box.cls[0].item())
                        name = yolo_names.get(cls, str(cls)) if yolo_names else str(cls)
                        draw_fancy_box(frame, x1, y1, x2, y2, (0, 200, 255))
                        put_tag(frame, name, (int(x1), int(y1) - 5))
            except Exception as e:
                put_tag(frame, f"YOLO erro: {str(e)[:25]}", (10, 70), bg=(0, 0, 255))

        # ====== DETECÇÃO DE FACE + EMOÇÃO ======
        if USE_FACES and face_det:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_det.process(rgb)
            if res.detections:
                for d in res.detections:
                    box = d.location_data.relative_bounding_box
                    x1, y1 = int(box.xmin * w), int(box.ymin * h)
                    x2, y2 = int((box.xmin + box.width) * w), int((box.ymin + box.height) * h)
                    draw_fancy_box(frame, x1, y1, x2, y2, (0, 255, 0))
                    if USE_EMOTION and frame_id % 6 == 0:
                        face_crop = frame[y1:y2, x1:x2]
                        emo = analyze_emotion(face_crop)
                        if emo:
                            put_tag(frame, f"Emoção: {emo}", (x1, y2 + 25), bg=(50, 50, 50))

        # ====== DETECÇÃO DE MÃOS ======
        if USE_HANDS and hands:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                for idx, lm in enumerate(res.multi_hand_landmarks):
                    mp_draw.draw_landmarks(frame, lm, mp_draw.HAND_CONNECTIONS)
                    dedos = fingers_state(lm.landmark, w, h)
                    name = gesture_name(dedos)
                    px, py = int(lm.landmark[8].x * w), int(lm.landmark[8].y * h)
                    put_tag(frame, name, (px - 40, py - 10), bg=(40, 40, 40))

        # ====== HUD ======
        put_tag(frame, f"FPS: {fps:.1f}", (10, 30))
        put_tag(frame, f"[o/f/e/h/q] Alternar | SRC: {src}", (10, h - 10))

        cv2.imshow("AI Vision", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('o'): USE_OBJECTS = not USE_OBJECTS
        elif k == ord('f'): USE_FACES = not USE_FACES
        elif k == ord('e'): USE_EMOTION = not USE_EMOTION
        elif k == ord('h'): USE_HANDS = not USE_HANDS

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    load_models()
    main()
