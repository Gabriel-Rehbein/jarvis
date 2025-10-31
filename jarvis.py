# ai_vision_live.py — ULTRA (Windows-friendly, ROI+Motion+Privacy+Logs)
import sys, os, time, argparse, csv, math
import cv2
import numpy as np

# =========================
# FLAGS (teclado)
# =========================
USE_OBJECTS = True
USE_FACES   = True
USE_EMOTION = True
USE_HANDS   = True
SHOW_HELP   = True
SHOW_GRID   = False
FACE_BLUR   = False
MOTION_ONLY = True   # YOLO só roda quando tiver movimento relevante
PAUSED      = False

# =========================
# Args / Config
# =========================
def parse_args():
    ap = argparse.ArgumentParser(
        description="AI Vision Live — YOLO + Face + Emotion + Hands (Windows-friendly)")
    ap.add_argument("--video", type=str, default=None, help="Arquivo de vídeo ou URL (RTSP/HTTP)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--conf", type=float, default=0.5, help="Confiança YOLO")
    ap.add_argument("--save-dir", type=str, default="runs/ai_vision")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--yolo-resize", type=int, default=0, help="Reduzir frame p/ YOLO (ex: 960). 0=desligado")
    ap.add_argument("--log", type=str, default="events.csv", help="Nome do CSV de eventos")
    ap.add_argument("--motion-sens", type=int, default=30, help="Sensibilidade de movimento (maior = menos sensível)")
    ap.add_argument("--roi-persist", action="store_true", help="Mantém ROI entre reaberturas de câmera")
    return ap.parse_args()

ARGS = parse_args()
os.makedirs(ARGS.save_dir, exist_ok=True)

# =========================
# Abrir fonte (arquivo/URL → DSHOW → MSMF)
# =========================
def is_url_like(path):
    return isinstance(path, str) and (path.startswith("rtsp://") or path.startswith("http://") or path.startswith("https://"))

def open_source(video_path=None, prefer_index=None):
    # 1) arquivo/URL explícito
    if video_path and (os.path.exists(video_path) or is_url_like(video_path)):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            print(f"[INFO] SRC:{video_path}")
            return cap, f"FILE:{video_path}"
        print(f"[WARN] Falha ao abrir: {video_path}")

    # 2) câmera por índice específico
    if isinstance(prefer_index, int):
        for api, name in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]:
            cap = cv2.VideoCapture(prefer_index, api)
            if cap.isOpened():
                print(f"[INFO] {name}:{prefer_index}")
                return cap, f"{name}:{prefer_index}"
            cap.release()

    # 3) varre 0..5
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
    TORCH_DEVICE = "cpu"
    if ARGS.device in ("auto","cuda"):
        try:
            import torch
            if torch.cuda.is_available():
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
        face_det_local = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        globals()["face_det"] = face_det_local
        print("[INFO] FaceDetection pronto.")
    except Exception as e:
        print("[WARN] FaceDetection indisponível:", e)

    # MediaPipe Hands
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands_local = mp_hands.Hands(False, 2, 0.7, 0.5)
        globals()["hands"] = hands_local
        globals()["mp_draw"] = mp.solutions.drawing_utils
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
        "Atalhos: [q] sair | [space] pausa | [n] +1 frame | [c] prox camera",
        "[o] objetos | [f] faces | [e] emoção | [h] mãos | [k] ajuda",
        "[b] blur faces | [g] grade | [m] motion-only YOLO on/off",
        "[r] gravar | [p] screenshot | Trackbar: conf YOLO",
        "Arraste com mouse p/ definir ROI (duplo clique p/ limpar)"
    ]
    y = 50
    for ln in lines:
        put_tag(img, ln, (10, y), bg=(0,0,0), fg=(255,255,255))
        y += 28

def draw_grid(img, w, h):
    thirds_x = [w//3, 2*w//3]; thirds_y = [h//3, 2*h//3]
    for x in thirds_x:
        cv2.line(img, (x,0), (x,h), (90,90,90), 1, cv2.LINE_AA)
    for y in thirds_y:
        cv2.line(img, (0,y), (w,y), (90,90,90), 1, cv2.LINE_AA)
    cv2.circle(img, (w//2, h//2), 6, (120,120,120), -1, cv2.LINE_AA)

# =========================
# Emoções (cooldown)
# =========================
_last_emo = {}
EMO_COOLDOWN = 0.6  # s

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
# Motion / ROI
# =========================
bg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=ARGS.motion_sens, detectShadows=False)
ROI = None
dragging = False
start_pt = (0,0)

def apply_roi(img):
    if ROI is None: return img, (0,0)
    (x1,y1,x2,y2) = ROI
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(img.shape[1]-1,x2), min(img.shape[0]-1,y2)
    if x2 <= x1 or y2 <= y1: return img, (0,0)
    return img[y1:y2, x1:x2], (x1,y1)

def on_mouse(event, x, y, flags, param):
    global ROI, dragging, start_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_pt = (x, y)
        ROI = None
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        ROI = (min(start_pt[0],x), min(start_pt[1],y), max(start_pt[0],x), max(start_pt[1],y))
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        if ROI and (ROI[2]-ROI[0] < 10 or ROI[3]-ROI[1] < 10):
            ROI = None
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        ROI = None

def draw_roi(img):
    if ROI is None: return
    (x1,y1,x2,y2) = ROI
    cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,0), 2, cv2.LINE_AA)
    put_tag(img, "ROI ativa (duplo-clique p/ limpar)", (x1, max(22, y1-6)), bg=(0,0,0))

# =========================
# Log CSV
# =========================
csv_path = os.path.join(ARGS.save_dir, ARGS.log)
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts","type","label","conf","x1","y1","x2","y2"])

def log_event(ev_type, label, conf, x1,y1,x2,y2):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), ev_type, label, f"{conf:.3f}", int(x1),int(y1),int(x2),int(y2)])

# =========================
# Utilidades
# =========================
def ensure_writer(out_path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # bom no Windows
    return cv2.VideoWriter(out_path, fourcc, max(fps, 1.0), size)

def setup_trackbars():
    cv2.namedWindow("AI Vision", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("YOLO conf x100", "AI Vision", int(ARGS.conf*100), 100, lambda _ : None)

def get_conf_from_trackbar():
    if cv2.getWindowProperty("AI Vision", 0) < 0: return ARGS.conf
    val = cv2.getTrackbarPos("YOLO conf x100", "AI Vision")
    return max(0.01, min(1.0, val/100.0))

def safe_reopen(cap, current_cam_index, video_path):
    cap.release(); time.sleep(0.2)
    if video_path:  # arquivo/URL
        return open_source(video_path)
    # ciclo de câmera
    next_idx = (current_cam_index + 1) % 6 if current_cam_index is not None else 0
    return open_source(None, prefer_index=next_idx), next_idx

# =========================
# MAIN
# =========================
def main():
    global USE_OBJECTS, USE_FACES, USE_EMOTION, USE_HANDS, SHOW_HELP, SHOW_GRID, FACE_BLUR, MOTION_ONLY, PAUSED

    load_models()
    cam_index = None
    cap, src = open_source(ARGS.video)
    if not cap: raise RuntimeError("Não foi possível abrir câmera/vídeo.")
    if src.startswith("DSHOW:") or src.startswith("MSMF:"):
        try: cam_index = int(src.split(":")[1])
        except: cam_index = None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ARGS.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ARGS.height)

    setup_trackbars()
    cv2.setMouseCallback("AI Vision", on_mouse)

    prev_t = time.time()
    fps_smooth = 0.0
    frame_id = 0
    writer = None
    recording = False
    last_fail = 0
    fail_thresh = 40

    while True:
        if not PAUSED:
            ok, frame = cap.read()
            if not ok:
                last_fail += 1
                if last_fail >= fail_thresh:
                    print("[WARN] Sem sinal. Tentando reabrir...")
                    (cap, src2), cam_index = safe_reopen(cap, cam_index, ARGS.video)
                    if not cap:
                        print("[ERRO] Reabertura falhou. Saindo.")
                        break
                    src = src2
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ARGS.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ARGS.height)
                    last_fail = 0
                    if not ARGS.roi_persist:
                        globals()["ROI"] = None
                continue
            last_fail = 0
            frame_id += 1
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        now = time.time()
        inst_fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        # média móvel exponencial p/ estabilidade
        fps_smooth = 0.9*fps_smooth + 0.1*inst_fps if fps_smooth > 0 else inst_fps

        # ====== ROI & MOTION ======
        view = frame
        offset = (0,0)
        if ROI is not None:
            view, offset = apply_roi(frame)

        motion_trigger = True
        if MOTION_ONLY:
            mask = bg.apply(view if not PAUSED else view*0)
            # ruído → morfologia
            mask = cv2.medianBlur(mask, 5)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area = sum(cv2.contourArea(c) for c in cnts)
            motion_trigger = area > (w*h*0.001)  # 0.1% da área do frame
            if SHOW_GRID:
                # mostrar área de movimento (pequeno overlay no canto)
                mini = cv2.resize(mask, (w//5, h//5))
                frame[0:h//5, w-w//5:w] = cv2.cvtColor(mini, cv2.COLOR_GRAY2BGR)

        # ====== OBJETOS (YOLO) ======
        if USE_OBJECTS and yolo_model is not None and motion_trigger and not PAUSED:
            try:
                conf_th = get_conf_from_trackbar()
                infer_img = view
                if ARGS.yolo_resize and ARGS.yolo_resize < infer_img.shape[1]:
                    scale = ARGS.yolo_resize / float(infer_img.shape[1])
                    infer_img = cv2.resize(infer_img, (int(infer_img.shape[1]*scale), int(infer_img.shape[0]*scale)), interpolation=cv2.INTER_AREA)

                results = yolo_model.predict(infer_img, conf=conf_th, verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        # remap ROI/resize
                        if infer_img is not view:
                            sx = view.shape[1]/float(infer_img.shape[1])
                            sy = view.shape[0]/float(infer_img.shape[0])
                            x1, x2 = x1*sx, x2*sx
                            y1, y2 = y1*sy, y2*sy
                        x1 += offset[0]; x2 += offset[0]
                        y1 += offset[1]; y2 += offset[1]

                        cls = int(box.cls[0].item()) if hasattr(box.cls[0],'item') else int(box.cls[0])
                        name = yolo_names.get(cls, str(cls)) if yolo_names else str(cls)
                        conf = float(box.conf[0]) if hasattr(box.conf[0],'__float__') else float(box.conf[0])
                        draw_fancy_box(frame, x1, y1, x2, y2, (0,200,255), 2)
                        put_tag(frame, f"{name} {conf:.2f}", (int(x1), max(22, int(y1)-6)))
                        log_event("object", name, conf, x1,y1,x2,y2)
            except Exception as e:
                put_tag(frame, f"YOLO err: {str(e)[:28]}", (10, 70), bg=(0,0,255))

        # ====== FACES + EMOÇÃO / BLUR ======
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

                    if FACE_BLUR:
                        face = frame[y1:y2, x1:x2]
                        if face.size > 0:
                            face = cv2.GaussianBlur(face, (31,31), 0)
                            frame[y1:y2, x1:x2] = face
                    else:
                        draw_fancy_box(frame, x1, y1, x2, y2, (0,255,0), 2)
                        put_tag(frame, "Face", (x1, max(22, y1-6)), bg=(0,128,0))

                    if USE_EMOTION and DeepFace is not None and (x2>x1 and y2>y1) and not FACE_BLUR:
                        key = _bbox_key(x1,y1,x2,y2)
                        emo = analyze_emotion(frame[y1:y2, x1:x2], key)
                        if emo:
                            put_tag(frame, f"Emoção: {emo}", (x1, y2 + 24), bg=(60,60,60))
                            log_event("emotion", emo, 1.0, x1,y1,x2,y2)

        # ====== MÃOS / GESTOS ======
        if USE_HANDS and hands is not None and not PAUSED:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
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
                    log_event("gesture", name, 1.0, px-20, py-20, px+20, py+20)

        # ====== HUD ======
        if SHOW_GRID: draw_grid(frame, w, h)
        draw_roi(frame)
        put_tag(frame, f"FPS: {fps_smooth:.1f}", (10, 30), bg=(0,0,0))
        status = f"[O:{'ON' if USE_OBJECTS else 'off'} F:{'ON' if USE_FACES else 'off'} E:{'ON' if USE_EMOTION else 'off'} H:{'ON' if USE_HANDS else 'off'}] {ARGS.width}x{ARGS.height} | SRC {src}"
        put_tag(frame, status, (10, h-10), bg=(0,0,0))
        if SHOW_HELP: draw_help(frame, w, h)

        cv2.imshow("AI Vision", frame)

        # gravação
        if recording:
            if writer is None:
                out_path = os.path.join(ARGS.save_dir, f"rec_{time.strftime('%Y%m%d_%H%M%S')}.avi")
                writer = ensure_writer(out_path, fps_smooth, (w, h))
                print(f"[REC] Gravando em: {out_path}")
            writer.write(frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord(' '): PAUSED = not PAUSED
        elif k == ord('n'):
            PAUSED = True
            ok, fr = cap.read()
            if ok: frame = cv2.flip(fr, 1)
        elif k == ord('o'): USE_OBJECTS = not USE_OBJECTS
        elif k == ord('f'): USE_FACES   = not USE_FACES
        elif k == ord('e'): USE_EMOTION = not USE_EMOTION
        elif k == ord('h'): USE_HANDS   = not USE_HANDS
        elif k == ord('k'): SHOW_HELP   = not SHOW_HELP
        elif k == ord('g'): SHOW_GRID   = not SHOW_GRID
        elif k == ord('b'): FACE_BLUR   = not FACE_BLUR
        elif k == ord('m'): MOTION_ONLY = not MOTION_ONLY
        elif k == ord('p'):
            path = os.path.join(ARGS.save_dir, f"shot_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(path, frame); print(f"[SHOT] {path}")
        elif k == ord('r'):
            recording = not recording
            if not recording and writer is not None:
                writer.release(); writer = None; print("[REC] Finalizado.")
        elif k == ord('c'):
            if ARGS.video is None:  # só faz sentido p/ webcam local
                (cap, src), cam_index = safe_reopen(cap, cam_index if cam_index is not None else -1, None)
                if not cap:
                    print("[ERRO] Não há mais câmeras."); break
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ARGS.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ARGS.height)

    if writer: writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
