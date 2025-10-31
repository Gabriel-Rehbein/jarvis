import cv2, time, threading

def open_any_camera(prefer=None):
    # 1) Preferido (se dado)
    if prefer is not None:
        for api, name in [(cv2.CAP_DSHOW,"DSHOW"), (cv2.CAP_MSMF,"MSMF")]:
            cap = cv2.VideoCapture(prefer, api)
            if cap.isOpened():
                return cap, f"{name}:{prefer}"
            cap.release()
    # 2) Varre 0..5 (DSHOW, depois MSMF)
    for api, name in [(cv2.CAP_DSHOW,"DSHOW"), (cv2.CAP_MSMF,"MSMF")]:
        for i in range(6):
            cap = cv2.VideoCapture(i, api)
            if cap.isOpened():
                return cap, f"{name}:{i}"
            cap.release()
    return None, None

class VideoCamera:
    def __init__(self, width=1280, height=720):
        self.cap, self.src_label = open_any_camera(None)
        if self.cap is None:
            raise RuntimeError("Nenhuma câmera disponível.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # estados/flags de UI
        self.objects     = True
        self.faces       = True
        self.emotion     = True
        self.hands       = True
        self.help        = True
        self.face_blur   = False
        self.grid        = False
        self.motion_only = True
        self.auto_rec    = False
        self.poly_mode     = False
        self.tripwire_mode = False
        self.poly_points   = []
        self.tripwire      = None

        self.paused    = False
        self.yolo_conf = 0.50
        self.roi       = None
        self.fps_val   = 0.0
        self.size_label= "-"

        # thread de captura (buffer de último frame)
        self._lock = threading.Lock()
        self._running = True
        self._last_frame = None
        self._reader = threading.Thread(target=self._capture_loop, daemon=True)
        self._reader.start()

    # ---------- captura em background ----------
    def _capture_loop(self):
        prev = time.time()
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                # pequena pausa e tenta seguir
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            self.size_label = f"{w}x{h}"
            now = time.time()
            self.fps_val = 1.0 / max(now - prev, 1e-6)
            prev = now

            # grid na visualização
            if self.grid:
                for x in [w//3, 2*w//3]:
                    cv2.line(frame, (x, 0), (x, h), (90, 90, 90), 1)
                for y in [h//3, 2*h//3]:
                    cv2.line(frame, (0, y), (w, y), (90, 90, 90), 1)

            # ROI (apenas visual no stream web)
            if self.roi:
                x1, y1, x2, y2 = self.roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            with self._lock:
                self._last_frame = frame
            # ~60fps máx stream; segura se muito rápido
            time.sleep(0.001)

    def get_frame(self):
        with self._lock:
            return None if self._last_frame is None else self._last_frame.copy()

    def get_jpeg(self):
        frame = self.get_frame()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return None if not ok else buf.tobytes()

    def mjpeg_generator(self):
        boundary = b"--frame"
        while True:
            jpg = self.get_jpeg()
            if jpg is None:
                time.sleep(0.01)
                continue
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

    # ---------- API ----------
    def health(self):
        f = self.get_frame()
        return (f is not None), (self.src_label if f is not None else "sem frame")

    def open_index(self, idx):
        # troca de câmera em runtime
        try:
            if idx is None:
                return False, "index ausente"
            new_cap, label = open_any_camera(idx)
            if new_cap is None:
                return False, f"não abriu index {idx}"
            with self._lock:
                old = self.cap
                self.cap = new_cap
                self.src_label = label
            try:
                old.release()
            except Exception:
                pass
            return True, f"ok: {label}"
        except Exception as e:
            return False, str(e)

    def set_conf(self, conf: float):
        self.yolo_conf = float(conf)

    def set_roi_norm(self, nx1, ny1, nx2, ny2):
        fr = self.get_frame()
        if fr is None:
            return
        h, w = fr.shape[:2]
        x1 = int(max(0, min(1, nx1)) * w)
        y1 = int(max(0, min(1, ny1)) * h)
        x2 = int(max(0, min(1, nx2)) * w)
        y2 = int(max(0, min(1, ny2)) * h)
        self.roi = None if (x2 <= x1 or y2 <= y1) else (x1, y1, x2, y2)

    def clear_roi(self):
        self.roi = None

    def action(self, cmd: str):
        if not cmd:
            return False, "missing cmd"
        if   cmd == "quit":          self._running = False
        elif cmd == "pause":         self.paused = not self.paused  # (opcional: pausar loop de IA)
        elif cmd == "next_frame":    pass
        elif cmd == "next_camera":   self._cycle_camera()
        elif cmd == "record_toggle": pass
        elif cmd == "screenshot":    pass
        elif cmd == "poly_mode":     self.poly_mode = not self.poly_mode
        elif cmd == "poly_clear":    self.poly_points = []
        elif cmd == "tripwire_mode": self.tripwire_mode = not self.tripwire_mode
        elif cmd == "toggle_objects":     self.objects = not self.objects
        elif cmd == "toggle_faces":       self.faces = not self.faces
        elif cmd == "toggle_emotion":     self.emotion = not self.emotion
        elif cmd == "toggle_hands":       self.hands = not self.hands
        elif cmd == "toggle_help":        self.help = not self.help
        elif cmd == "toggle_face_blur":   self.face_blur = not self.face_blur
        elif cmd == "toggle_grid":        self.grid = not self.grid
        elif cmd == "toggle_motion_only": self.motion_only = not self.motion_only
        elif cmd == "toggle_auto_rec":    self.auto_rec = not self.auto_rec
        else:
            return False, f"unknown cmd: {cmd}"
        return True, self.status()

    def _cycle_camera(self):
        # tenta próxima disponível
        cur = self.src_label
        # simples: tenta índices 0..5 até achar outro diferente
        for i in range(6):
            ok, msg = self.open_index(i)
            if ok and self.src_label != cur:
                return

    def status(self):
        return {
            "src": self.src_label,
            "size": self.size_label,
            "fps": self.fps_val,
            "objects": self.objects,
            "faces": self.faces,
            "emotion": self.emotion,
            "hands": self.hands,
            "help": self.help,
            "face_blur": self.face_blur,
            "grid": self.grid,
            "motion_only": self.motion_only,
            "auto_rec": self.auto_rec,
            "poly_mode": self.poly_mode,
            "tripwire_mode": self.tripwire_mode,
            "poly_points": len(self.poly_points),
            "tripwire": self.tripwire is not None,
            "roi": None if not self.roi else {
                "x1": self.roi[0], "y1": self.roi[1],
                "x2": self.roi[2], "y2": self.roi[3],
            },
            "yolo_conf": self.yolo_conf,
        }

    def stop(self):
        self._running = False
        time.sleep(0.05)
        try:
            self.cap.release()
        except Exception:
            pass
