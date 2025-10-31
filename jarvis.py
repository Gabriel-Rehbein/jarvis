# ai_vision_live.py — PRO (Windows-friendly)
import sys, os, time, argparse, uuid, math
import cv2
import numpy as np

# =========================
# FLAGS padrão (teclado)
# =========================
USE_OBJECTS = True
USE_FACES   = True
USE_EMOTION = True
USE_HANDS   = True
SHOW_HELP   = True

# =========================
# Args / Config
# =========================
def parse_args():
    ap = argparse.ArgumentParser(
        description="AI Vision Live — YOLO + Face + Emotion + Hands (Windows-friendly)")
    ap.add_argument("--video", type=str, default=None, help="Arquivo de vídeo")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--conf", type=float, default=0.5, help="Confiança YOLO")
    ap.add_argument("--save-dir", type=str, default="runs/ai_vision")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--yolo-resize", type=int, default=0, help="Reduzir frame p/ YOLO (ex: 960). 0=desligado")
    return ap.parse_args()

ARGS = parse_args()
os.makedirs(ARGS.save_dir, exist_ok=True)

# =========================
# Abrir fonte (arquivo → DSHOW → MSMF)
# =========================
def open_source(video_path=None):
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            print(f"[INFO] FILE:{video_path}")
            return cap, f"FILE:{video_path}"
        print(f"[WARN] Falha ao abrir arquivo: {video_path}")

    for api, name in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]:
        for i in range(6):
            cap = cv2.VideoCapture(i, api)
            if cap.isOpened():
                print(f"[INFO] {name}:{i}")
                return cap, f"{name}:{i}"
            cap.release()
    return None, None

# =========================
# Modelos
# =========================
yolo_model = None
yolo_names = None
face_det   = None
hands      = None
mp_draw    = None
DeepFace   = None
TORCH_DEVICE = "cpu"

def load_models():
    global yolo_model, yolo_names, face_det, hands, mp_draw, DeepFace, TORCH_DEVICE
    # Device
    TORCH_DEVICE = "cpu"
    if ARGS.device in ("auto","cuda"):
        try:
            import torch
            if (ARGS.device == "cuda" and torch.cuda.is_available()) or (ARGS.device=="auto" and torch.cuda.is_available()):
                TORCH_DEVICE = "cuda"
        except Exception:
            pass
    print(f"[INFO] Device: {TORCH_DEVICE}")

    # YOLO
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")
        try:
            if TORCH_DEVICE == "cuda":
                yolo_model.to("cuda")
        except Exception:
            pass
        yolo_names = yolo_model.names
        print("[INFO] YOLO pronto.")
    except Exception as e:
        print("[WARN] YOLO indisponível:", e)

    # MediaPipe Face
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        face_det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        print("[INFO] FaceDetection pronto.")
    except Exception as e:
        print("[WARN] FaceDetection indisponível:", e)

    # MediaPipe Hands
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        # (static_image_mode, max_num_hands, min_detection_confidence, min_tracking_confidence)
        hands = mp_hands.Hands(False, 2, 0.7, 0.5)
        mp_draw = mp.solutions.drawing_utils
        print("[INFO] Hands pronto.")
    except Exception as e:
        print("[WARN] Hands indisponível:", e)

    # DeepFace
    try:
        from deepface import DeepFace as _DF
        globals()["DeepFace"] = _DF
        print("[INFO] DeepFace pronto.")
    except Exception as e:
        print("[WARN] DeepFace indisponível:", e)

# =========================
# Desenho / UI
# =========================
def draw_fancy_box(img, x1, y1, x2, y2, color=(0, 255, 0), thick=2):
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
    for (px, py) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.circle(img, (px, py), 5, color, -1, cv2.LINE_AA)

def put_tag(img, text, org, bg=(0,0,0), fg=(255,255,255), scale=0.55, thick=2):
    x, y = org
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y + 8), bg, -1)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)

def draw_help(img, w, h):
    lines = [
        "Atalhos: [q] sair | [space] pausa | [n] +1 frame",
        "[o] objetos | [f] faces | [e] emoção | [h] mãos | [k] ajuda",
        "[r] gravar video | [p] screenshot | Trackbar: conf YOLO",
    ]
    y = 50
    for ln in lines:
        put_tag(img, ln, (10, y), bg=(0,0,0), fg=(255,255,255))
        y += 28

# =========================
# Emoções (cooldown)
# =========================
# Guardamos timestamp do último cálculo por face (bbox hash simples)
_last_emo = {}
EMO_COOLDOWN = 0.6  # segundos

def _bbox_key(x1,y1,x2,y2):
    return f"{int(x1/10)}-{int(y1/10)}-{int(x2/10)}-{int(y2/10)}"

def analyze_emotion(face_bgr, key):
    if DeepFace is None: return ""
    now = time.time()
    last = _last_emo.get(key, {"t":0, "emo":""})
    if now - last["t"] < EMO_COOLDOWN:
        return last["emo"]
    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        res = DeepFace.analyze(face_rgb, actions=["emotion"], enforce_detection=False, prog_bar=False)
        if isinstance(res, list) and res:
            res = res[0]
        emo = res.get("dominant_emotion", last["emo"])
        _last_emo[key] = {"t": now, "emo": emo}
        return emo
    except Exception:
        return last["emo"]

# =========================
# Gestos
# =========================
def fingers_state(lm, w, h, handed="Right"):
    thumb_tip_x, thumb_ip_x = lm[4].x * w, lm[3].x * w
    dedos = [1 if (thumb_tip_x > thumb_ip_x if handed=="Right" else thumb_tip_x < thumb_ip_x) else 0]
    for t, p in zip([8,12,16,20],[6,10,14,18]):
        dedos.append(1 if lm[t].y < lm[p].y else 0)
    return dedos

def gesture_name(dedos):
    s = sum(dedos)
    if dedos == [1,0,0,0,0]: return "JOINHA"
    if dedos == [0,1,1,0,0]: return "V (PAZ)"
    if s == 5: return "ABERTA"
    if s == 0: return "PUNHO"
    return f"{s} dedos"

# =========================
# Utilidades
# =========================
def ensure_writer(out_path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # bom no Windows
    return cv2.VideoWriter(out_path, fourcc, max(fps, 1.0), size)

def safe_reopen(cap, src_label, video_path=None):
    # tenta reabrir a cada chamada
    cap.release()
    time.sleep(0.2)
    return open_source(video_path if video_path else None)

# =========================
# Trackbars (conf YOLO)
# =========================
def on_trackbar(_): pass
def setup_trackbars():
    cv2.namedWindow("AI Vision", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("YOLO conf x100", "AI Vision", int(ARGS.conf*100), 100, on_trackbar)

def get_conf_from_trackbar():
    val = cv2.getTrackbarPos("YOLO conf x100", "AI Vision") if cv2.getWindowProperty("AI Vision", 0) >= 0 else int(ARGS.conf*100)
    return max(0.01, min(1.0, val/100.0))

# =========================
# MAIN
# =========================
def main():
    global USE_OBJECTS, USE_FACES, USE_EMOTION, USE_HANDS, SHOW_HELP, yolo_model
    load_models()

    cap, src = open_source(ARGS.video)
    if not cap: raise RuntimeError("Não foi possível abrir câmera/vídeo.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ARGS.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ARGS.height)

    setup_trackbars()
    prev_t = time.time()
    frame_id = 0
    paused = False
    writer = None
    recording = False
    last_fail = 0
    fail_thresh = 40  # frames sem sinal → reabrir

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                last_fail += 1
                if last_fail >= fail_thresh:
                    print("[WARN] Sem sinal. Tentando reabrir...")
                    cap, src = safe_reopen(cap, src, ARGS.video)
                    if not cap:
                        print("[ERRO] Reabertura falhou. Saindo.")
                        break
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ARGS.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ARGS.height)
                    last_fail = 0
                continue
            last_fail = 0
            frame_id += 1
        else:
            # quando pausado, apenas redesenha última tela
            pass

        if not paused:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now

        # ============== OBJETOS (YOLO) ==============
        if USE_OBJECTS and yolo_model is not None:
            try:
                conf_th = get_conf_from_trackbar()
                infer_img = frame
                if ARGS.yolo_resize and ARGS.yolo_resize < w:
                    scale = ARGS.yolo_resize / float(w)
                    infer_img = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

                results = yolo_model.predict(infer_img, conf=conf_th, verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        # remap se redimensionou
                        if infer_img is not frame:
                            sx, sy = w/float(infer_img.shape[1]), h/float(infer_img.shape[0])
                            x1, x2 = x1*sx, x2*sx
                            y1, y2 = y1*sy, y2*sy
                        cls = int(box.cls[0].item()) if hasattr(box.cls[0],'item') else int(box.cls[0])
                        name = yolo_names.get(cls, str(cls)) if yolo_names else str(cls)
                        draw_fancy_box(frame, x1, y1, x2, y2, (0,200,255), 2)
                        put_tag(frame, f"{name} {float(box.conf[0]):.2f}", (int(x1), max(22, int(y1)-6))), 
            except Exception as e:
                put_tag(frame, f"YOLO err: {str(e)[:28]}", (10, 70), bg=(0,0,255))

        # ============== FACES + EMOÇÃO ==============
        if USE_FACES and face_det is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = face_det.process(rgb)
            if out and out.detections:
                for det in out.detections:
                    box = det.location_data.relative_bounding_box
                    x1 = int(box.xmin * w); y1 = int(box.ymin * h)
                    x2 = int((box.xmin + box.width)  * w)
                    y2 = int((box.ymin + box.height) * h)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w-1, x2), min(h-1, y2)
                    draw_fancy_box(frame, x1, y1, x2, y2, (0,255,0), 2)
                    put_tag(frame, "Face", (x1, max(22, y1-6)), bg=(0,128,0))

                    if USE_EMOTION and DeepFace is not None and (x2>x1 and y2>y1):
                        key = _bbox_key(x1,y1,x2,y2)
                        face_crop = frame[y1:y2, x1:x2]
                        emo = analyze_emotion(face_crop, key)
                        if emo:
                            put_tag(frame, f"Emoção: {emo}", (x1, y2 + 24), bg=(60,60,60))

        # ============== MÃOS / GESTOS ==============
        if USE_HANDS and hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                # tentar handedness
                handed = []
                try:
                    handed = [h.classification[0].label for h in res.multi_handedness]
                except Exception:
                    handed = ["Right"] * len(res.multi_hand_landmarks)

                for i, lm in enumerate(res.multi_hand_landmarks):
                    mp_draw.draw_landmarks(frame, lm, mp_draw.HAND_CONNECTIONS)
                    dedos = fingers_state(lm.landmark, w, h, handed[i] if i < len(handed) else "Right")
                    name = gesture_name(dedos)
                    px = int(lm.landmark[8].x * w)
                    py = int(lm.landmark[8].y * h)
                    put_tag(frame, name, (max(0, px-50), max(0, py-12)), bg=(40,40,40))

        # ============== HUD / HELP ==============
        put_tag(frame, f"FPS: {fps:.1f}", (10, 30), bg=(0,0,0))
        status = f"[O:{'ON' if USE_OBJECTS else 'off'} F:{'ON' if USE_FACES else 'off'} E:{'ON' if USE_EMOTION else 'off'} H:{'ON' if USE_HANDS else 'off'}] {ARGS.width}x{ARGS.height} | SRC {src}"
        put_tag(frame, status, (10, h-10), bg=(0,0,0))
        if SHOW_HELP: draw_help(frame, w, h)

        cv2.imshow("AI Vision", frame)

        # gravação
        if recording:
            if writer is None:
                out_path = os.path.join(ARGS.save_dir, f"rec_{time.strftime('%Y%m%d_%H%M%S')}.avi")
                writer = ensure_writer(out_path, fps, (w, h))
                print(f"[REC] Gravando em: {out_path}")
            writer.write(frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord(' '):  # pause
            paused = not paused
        elif k == ord('n'):  # next frame
            paused = True
            ok, fr = cap.read()
            if ok: frame = cv2.flip(fr, 1)
        elif k == ord('o'): USE_OBJECTS = not USE_OBJECTS
        elif k == ord('f'): USE_FACES   = not USE_FACES
        elif k == ord('e'): USE_EMOTION = not USE_EMOTION
        elif k == ord('h'): USE_HANDS   = not USE_HANDS
        elif k == ord('k'): SHOW_HELP   = not SHOW_HELP
        elif k == ord('p'):
            path = os.path.join(ARGS.save_dir, f"shot_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(path, frame)
            print(f"[SHOT] {path}")
        elif k == ord('r'):
            recording = not recording
            if not recording and writer is not None:
                writer.release()
                writer = None
                print("[REC] Finalizado.")

    if writer: writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
