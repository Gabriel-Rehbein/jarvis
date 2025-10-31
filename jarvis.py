# ai_vision_live.py — ULTRA++ (Tripwire, Zona Poligonal, Filtros de Classe, Dwell & Alertas)
import sys, os, time, argparse, csv, json, math, platform
import cv2
import numpy as np
from collections import OrderedDict, defaultdict

# ======= FLAGS =======
USE_OBJECTS = True
USE_FACES   = True
USE_EMOTION = True
USE_HANDS   = True
SHOW_HELP   = True
SHOW_GRID   = False
FACE_BLUR   = False
MOTION_ONLY = True
PAUSED      = False

# ======= ARGS =======
def parse_args():
    ap = argparse.ArgumentParser(description="AI Vision Live — ULTRA++")
    ap.add_argument("--video", type=str, default=None, help="Arquivo/URL (RTSP/HTTP)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--conf", type=float, default=0.5, help="Confiança YOLO")
    ap.add_argument("--save-dir", type=str, default="runs/ai_vision")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--yolo-resize", type=int, default=0, help="Reduz frame p/ YOLO (ex: 960). 0=off")
    ap.add_argument("--log", type=str, default="events.csv", help="CSV de eventos")
    ap.add_argument("--motion-sens", type=int, default=30, help="Sensibilidade movimento (↑=menos sensível)")
    ap.add_argument("--roi-persist", action="store_true", help="Mantém ROI entre reaberturas")
    ap.add_argument("--auto-rec", action="store_true", help="Gravar auto por movimento")
    ap.add_argument("--rec-cooldown", type=float, default=3.0, help="s sem movimento p/ parar")
    ap.add_argument("--rec-mp4", action="store_true", help="Salvar em MP4 (mp4v)")
    ap.add_argument("--allow", type=str, default="", help="Classes permitidas (csv). vazio = todas")
    ap.add_argument("--deny",  type=str, default="", help="Classes negadas (csv)")
    ap.add_argument("--alert-classes", type=str, default="", help="Classes que disparam alerta")
    ap.add_argument("--screenshot-on-alert", action="store_true", help="Salvar foto em alerta")
    ap.add_argument("--dwell-alert", type=float, default=0.0, help="Segundos p/ alertar permanência por ID (0 desliga)")
    ap.add_argument("--dump-json", action="store_true", help="Salvar JSON por frame com anotações")
    return ap.parse_args()

ARGS = parse_args()
os.makedirs(ARGS.save_dir, exist_ok=True)

ALLOW = set([c.strip().lower() for c in ARGS.allow.split(",") if c.strip()])
DENY  = set([c.strip().lower() for c in ARGS.deny.split(",") if c.strip()])
ALERT_CLASSES = set([c.strip().lower() for c in ARGS.alert_classes.split(",") if c.strip()])

# ======= Fonte =======
def is_url_like(path): return isinstance(path, str) and path.startswith(("rtsp://","http://","https://"))
def open_source(video_path=None, prefer_index=None):
    if video_path and (os.path.exists(video_path) or is_url_like(video_path)):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened(): return cap, f"FILE:{video_path}"
    if isinstance(prefer_index, int):
        for api,name in [(cv2.CAP_DSHOW,"DSHOW"),(cv2.CAP_MSMF,"MSMF")]:
            cap = cv2.VideoCapture(prefer_index, api)
            if cap.isOpened(): return cap, f"{name}:{prefer_index}"
            cap.release()
    for api,name in [(cv2.CAP_DSHOW,"DSHOW"),(cv2.CAP_MSMF,"MSMF")]:
        for i in range(6):
            cap = cv2.VideoCapture(i, api)
            if cap.isOpened(): return cap, f"{name}:{i}"
            cap.release()
    return None, None

# ======= Modelos =======
yolo_model = None; yolo_names = None
face_det = None; hands = None; mp_draw = None; DeepFace = None
TORCH_DEVICE = "cpu"; TORCH_HALF = False

def load_models():
    global yolo_model,yolo_names,face_det,hands,mp_draw,DeepFace,TORCH_DEVICE,TORCH_HALF
    TORCH_DEVICE = "cpu"
    if ARGS.device in ("auto","cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                TORCH_DEVICE = "cuda"
                try:
                    torch.set_float32_matmul_precision("high")
                    TORCH_HALF = True
                except Exception: TORCH_HALF = False
        except Exception: pass
    print(f"[INFO] Device: {TORCH_DEVICE} | half:{TORCH_HALF}")

    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")
        try:
            if TORCH_DEVICE == "cuda": yolo_model.to("cuda")
        except Exception: pass
        yolo_names = yolo_model.names
        print("[INFO] YOLO pronto.")
    except Exception as e:
        print("[WARN] YOLO indisponível:", e)

    try:
        import mediapipe as mp
        face_det_local = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        globals()["face_det"] = face_det_local
        mp_hands = mp.solutions.hands
        globals()["hands"] = mp_hands.Hands(False, 2, 0.7, 0.5)
        globals()["mp_draw"] = mp.solutions.drawing_utils
        print("[INFO] MediaPipe pronto.")
    except Exception as e:
        print("[WARN] MediaPipe indisponível:", e)

    try:
        from deepface import DeepFace as _DF
        globals()["DeepFace"] = _DF
        print("[INFO] DeepFace pronto.")
    except Exception as e:
        print("[WARN] DeepFace indisponível:", e)

# ======= UI =======
CLASS_COLORS = defaultdict(lambda: (0,200,255))
CLASS_COLORS.update({
    "person": (60,220,60), "cell phone": (60,120,255), "laptop": (40,200,200),
    "bottle": (200,160,60), "car": (80,80,255), "dog": (200,60,120)
})
def draw_fancy_box(img, x1,y1,x2,y2, color=(0,255,0), thick=2):
    x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
    cv2.rectangle(img,(x1,y1),(x2,y2),color,thick,cv2.LINE_AA)
    for (px,py) in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
        cv2.circle(img,(px,py),5,color,-1,cv2.LINE_AA)

def put_tag(img, text, org, bg=(0,0,0), fg=(255,255,255), scale=0.55, thick=2):
    x,y = org
    (tw,th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img,(x, y-th-8),(x+tw+10, y+8), bg, -1)
    cv2.putText(img, text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)

def draw_help(img, w, h):
    lines = [
        "Atalhos: [q] sair | [space] pausa | [n] +1 | [c] prox cam",
        "[o] objs | [f] faces | [e] emoção | [h] mãos | [k] ajuda",
        "[b] blur faces | [g] grade | [m] motion-only | [a] auto-rec",
        "[t] tripwire (2 pts) | [z] polígono | [x] limpar polígono",
        "[r] gravar | [p] screenshot | Trackbar: conf YOLO | Mouse: ROI"
    ]
    y = 50
    for ln in lines:
        put_tag(img, ln, (10,y), bg=(0,0,0)); y += 28

def draw_grid(img, w, h):
    for x in [w//3, 2*w//3]: cv2.line(img,(x,0),(x,h),(90,90,90),1,cv2.LINE_AA)
    for y in [h//3, 2*h//3]: cv2.line(img,(0,y),(w,y),(90,90,90),1,cv2.LINE_AA)
    cv2.circle(img,(w//2,h//2),6,(120,120,120),-1,cv2.LINE_AA)

# ======= Emoção =======
_last_emo = {}; EMO_COOLDOWN = 0.6
def _bbox_key(x1,y1,x2,y2): return f"{int(x1/10)}-{int(y1/10)}-{int(x2/10)}-{int(y2/10)}"
def analyze_emotion(face_bgr, key):
    if DeepFace is None: return ""
    now = time.time(); last = _last_emo.get(key, {"t":0,"emo":""})
    if now - last["t"] < EMO_COOLDOWN: return last["emo"]
    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        res = DeepFace.analyze(face_rgb, actions=["emotion"], enforce_detection=False, prog_bar=False)
        if isinstance(res, list) and res: res = res[0]
        emo = res.get("dominant_emotion", last["emo"])
        _last_emo[key] = {"t": now, "emo": emo}
        return emo
    except Exception: return last["emo"]

# ======= Gestos =======
def fingers_state(lm, w,h, handed="Right"):
    thumb_tip_x, thumb_ip_x = lm[4].x*w, lm[3].x*w
    dedos = [1 if (thumb_tip_x > thumb_ip_x if handed=="Right" else thumb_tip_x < thumb_ip_x) else 0]
    for t,p in zip([8,12,16,20],[6,10,14,18]): dedos.append(1 if lm[t].y < lm[p].y else 0)
    return dedos
def gesture_name(d): s=sum(d); 
# (python exige bloco) manter simples:
def gesture_name(d):
    s = sum(d)
    if d == [1,0,0,0,0]: return "JOINHA"
    if d == [0,1,1,0,0]: return "V (PAZ)"
    if s == 5: return "ABERTA"
    if s == 0: return "PUNHO"
    return f"{s} dedos"

# ======= ROI / Motion =======
bg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=ARGS.motion_sens, detectShadows=False)
ROI = None; dragging = False; start_pt = (0,0)
POLY = []        # pontos do polígono de inclusão
POLY_MODE = False
TRIPWIRE = None  # ((x1,y1),(x2,y2))
TW_MODE = False

def apply_roi(img):
    if ROI is None: return img, (0,0)
    x1,y1,x2,y2 = ROI
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(img.shape[1]-1,x2), min(img.shape[0]-1,y2)
    if x2<=x1 or y2<=y1: return img, (0,0)
    return img[y1:y2, x1:x2], (x1,y1)

def point_in_poly(x,y, poly):
    if not poly: return True
    res=False; j=len(poly)-1
    for i in range(len(poly)):
        xi,yi = poly[i]; xj,yj = poly[j]
        if ((yi>y)!=(yj>y)) and (x < (xj-xi)*(y-yi)/(yj-yi+1e-9)+xi): res = not res
        j=i
    return res

def seg_intersect(p1,p2,p3,p4):
    def ccw(a,b,c): return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return (ccw(p1,p3,p4)!=ccw(p2,p3,p4)) and (ccw(p1,p2,p3)!=ccw(p1,p2,p4))

def on_mouse(event,x,y,flags,param):
    global ROI,dragging,start_pt,POLY,POLY_MODE,TRIPWIRE,TW_MODE
    if POLY_MODE:
        if event==cv2.EVENT_LBUTTONDOWN: POLY.append((x,y))
        elif event==cv2.EVENT_RBUTTONDOWN: POLY_MODE=False
        return
    if TW_MODE:
        if event==cv2.EVENT_LBUTTONDOWN:
            if TRIPWIRE is None: TRIPWIRE = [(x,y)]
            elif len(TRIPWIRE)==1: TRIPWIRE.append((x,y)); TW_MODE=False
            else: TRIPWIRE=[(x,y)]
        return
    if event==cv2.EVENT_LBUTTONDOWN:
        dragging=True; start_pt=(x,y); ROI=None
    elif event==cv2.EVENT_MOUSEMOVE and dragging:
        ROI=(min(start_pt[0],x), min(start_pt[1],y), max(start_pt[0],x), max(start_pt[1],y))
    elif event==cv2.EVENT_LBUTTONUP:
        dragging=False
        if ROI and (ROI[2]-ROI[0]<10 or ROI[3]-ROI[1]<10): ROI=None
    elif event==cv2.EVENT_LBUTTONDBLCLK:
        ROI=None

def draw_shapes(frame):
    if ROI is not None:
        x1,y1,x2,y2 = ROI
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2,cv2.LINE_AA)
        put_tag(frame,"ROI ativa (duplo-clique limpa)",(x1,max(22,y1-6)))
    if POLY:
        pts=np.array(POLY,np.int32)
        cv2.polylines(frame,[pts],False,(255,0,180),2,cv2.LINE_AA)
        if POLY_MODE: put_tag(frame,"Polígono: clique p/ pontos, direito p/ fechar",(10,80))
    if TRIPWIRE and len(TRIPWIRE)==2:
        cv2.line(frame,TRIPWIRE[0],TRIPWIRE[1],(0,0,255),2,cv2.LINE_AA)
        put_tag(frame,"Tripwire ativo",(min(TRIPWIRE[0][0],TRIPWIRE[1][0])+6,
                                       min(TRIPWIRE[0][1],TRIPWIRE[1][1])-6))

# ======= CSV / JSON / ALERT =======
csv_path = os.path.join(ARGS.save_dir, ARGS.log)
if not os.path.exists(csv_path):
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(["ts","type","id","label","conf","x1","y1","x2","y2","extra"])

def log_event(ev_type, tid, label, conf, x1,y1,x2,y2, extra=""):
    with open(csv_path,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), ev_type, tid, label, f"{conf:.3f}",
                                int(x1),int(y1),int(x2),int(y2), extra])

def beep():
    if platform.system().lower().startswith("win"):
        try:
            import winsound; winsound.Beep(1200, 110)
        except Exception: pass

def screenshot(frame, tag="alert"):
    path=os.path.join(ARGS.save_dir, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(path, frame); print(f"[SHOT] {path}")

def dump_json(frame_idx, w,h, objs, faces, hands_g, extras):
    if not ARGS.dump_json: return
    data = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame": frame_idx, "size": [w,h],
        "objects": [{"id":o["id"],"label":o["label"],"conf":o["conf"],"bbox":o["bbox"]} for o in objs],
        "faces": [{"bbox":f["bbox"],"emotion":f.get("emotion","") } for f in faces],
        "hands": [{"bbox":g["bbox"],"gesture":g["gesture"]} for g in hands_g],
        "extras": extras
    }
    p=os.path.join(ARGS.save_dir, f"frame_{frame_idx:08d}.json")
    with open(p,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False)

# ======= Tracking =======
class CentroidTracker:
    def __init__(self, max_dis=70, max_miss=25): 
        self.next_id=1; self.objects=OrderedDict(); self.bboxes=OrderedDict(); self.missed=OrderedDict()
        self.first_seen=OrderedDict(); self.last_center=OrderedDict()
        self.max_dis=max_dis; self.max_miss=max_miss
    def _centroid(self, box):
        x1,y1,x2,y2=box; return ((x1+x2)/2.0,(y1+y2)/2.0)
    def update(self, detections):
        # detections: [(x1,y1,x2,y2,label,conf)]
        if len(self.objects)==0:
            for d in detections:
                box=d[:4]; cid=self.next_id; self.next_id+=1
                self.objects[cid]=self._centroid(box); self.bboxes[cid]=box; self.missed[cid]=0
                self.first_seen[cid]=time.time(); self.last_center[cid]=self.objects[cid]
            return {cid:self.bboxes[cid] for cid in self.bboxes}
        obj_ids=list(self.objects.keys()); obj_pts=np.array([self.objects[i] for i in obj_ids],dtype="float32")
        det_boxes=[d[:4] for d in detections]
        det_pts=np.array([self._centroid(b) for b in det_boxes],dtype="float32") if det_boxes else np.zeros((0,2))
        if len(det_boxes)==0:
            for i in obj_ids:
                self.missed[i]+=1
                if self.missed[i]>self.max_miss:
                    for dct in (self.objects,self.bboxes,self.missed,self.first_seen,self.last_center):
                        dct.pop(i, None)
            return {cid:self.bboxes[cid] for cid in self.bboxes}
        D = np.linalg.norm(obj_pts[:,None,:]-det_pts[None,:,:], axis=2) if len(obj_pts) and len(det_pts) else np.zeros((0,0))
        used_obj=set(); used_det=set(); matches=[]
        if D.size>0:
            while True:
                i,j=np.unravel_index(np.argmin(D), D.shape)
                if i in used_obj or j in used_det: break
                if D[i,j]>self.max_dis: break
                used_obj.add(i); used_det.add(j); matches.append((obj_ids[i], det_boxes[j]))
                D[i,:]=1e9; D[:,j]=1e9
        for oid,box in matches:
            self.objects[oid]=self._centroid(box); self.bboxes[oid]=box; self.missed[oid]=0
        for idx,i in enumerate(obj_ids):
            if idx not in used_obj:
                self.missed[i]+=1
                if self.missed[i]>self.max_miss:
                    for dct in (self.objects,self.bboxes,self.missed,self.first_seen,self.last_center):
                        dct.pop(i,None)
        for k,box in enumerate(det_boxes):
            if k not in used_det:
                cid=self.next_id; self.next_id+=1
                self.objects[cid]=self._centroid(box); self.bboxes[cid]=box; self.missed[cid]=0
                self.first_seen[cid]=time.time(); self.last_center[cid]=self.objects[cid]
        return {cid:self.bboxes[cid] for cid in self.bboxes}

# ======= Util =======
def ensure_writer(path, fps, size):
    w,h=size
    if ARGS.rec_mp4:
        fourcc=cv2.VideoWriter_fourcc(*"mp4v")
        out=cv2.VideoWriter(path, fourcc, max(fps,1.0), (w,h))
        if out.isOpened(): return out
        print("[WARN] MP4 indisponível. Fallback AVI.")
    fourcc=cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(path.replace(".mp4",".avi"), fourcc, max(fps,1.0), (w,h))

def setup_trackbars():
    cv2.namedWindow("AI Vision", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("YOLO conf x100", "AI Vision", int(ARGS.conf*100), 100, lambda _ : None)
def get_conf_from_trackbar():
    if cv2.getWindowProperty("AI Vision",0) < 0: return ARGS.conf
    return max(0.01, min(1.0, cv2.getTrackbarPos("YOLO conf x100","AI Vision")/100.0))
def safe_reopen(cap, idx, video_path):
    cap.release(); time.sleep(0.2)
    if video_path: return open_source(video_path)
    next_idx=(idx+1)%6 if idx is not None else 0
    return open_source(None, prefer_index=next_idx), next_idx

# ======= MAIN =======
def main():
    global USE_OBJECTS, USE_FACES, USE_EMOTION, USE_HANDS, SHOW_HELP, SHOW_GRID, FACE_BLUR, MOTION_ONLY, PAUSED, POLY_MODE, TW_MODE
    load_models()
    cam_index=None
    cap, src = open_source(ARGS.video)
    if not cap: raise RuntimeError("Não foi possível abrir câmera/vídeo.")
    if src.startswith(("DSHOW:","MSMF:")):
        try: cam_index=int(src.split(":")[1])
        except: cam_index=None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ARGS.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ARGS.height)
    setup_trackbars()
    cv2.setMouseCallback("AI Vision", on_mouse)

    # salva config da sessão
    with open(os.path.join(ARGS.save_dir,"session.json"),"w") as f:
        json.dump({k:getattr(ARGS,k) for k in vars(ARGS)}, f, indent=2)

    prev_t=time.time(); fps_smooth=0.0; frame_id=0
    writer=None; recording=False; last_motion_ts=0.0
    last_fail=0; fail_thresh=40
    tracker=CentroidTracker(70,25)

    while True:
        if not PAUSED:
            ok, frame = cap.read()
            if not ok:
                last_fail+=1
                if last_fail>=fail_thresh:
                    print("[WARN] Sem sinal. Reabrindo...")
                    (cap,src2), cam_index = safe_reopen(cap, cam_index, ARGS.video)
                    if not cap: print("[ERRO] Reabertura falhou."); break
                    src=src2; cap.set(cv2.CAP_PROP_FRAME_WIDTH,ARGS.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,ARGS.height)
                    last_fail=0
                    if not ARGS.roi_persist: globals()["ROI"]=None
                continue
            last_fail=0; frame_id+=1; frame=cv2.flip(frame,1)

        h,w = frame.shape[:2]
        now=time.time(); inst_fps=1.0/max(now-prev_t,1e-6); prev_t=now
        fps_smooth = 0.9*fps_smooth + 0.1*inst_fps if fps_smooth>0 else inst_fps

        view,offset = (frame,(0,0)) if ROI is None else apply_roi(frame)

        # ===== MOTION =====
        motion_trigger=True
        if MOTION_ONLY:
            mask = bg.apply(view if not PAUSED else view*0)
            mask = cv2.medianBlur(mask,5); _,mask=cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
            cnts,_=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area=sum(cv2.contourArea(c) for c in cnts)
            motion_trigger = area > (w*h*0.001)
            if motion_trigger: last_motion_ts = now

        # ===== OBJETOS =====
        dets=[]
        if USE_OBJECTS and yolo_model is not None and motion_trigger and not PAUSED:
            try:
                conf_th=get_conf_from_trackbar()
                infer_img=view
                if ARGS.yolo_resize and ARGS.yolo_resize<infer_img.shape[1]:
                    scale=ARGS.yolo_resize/float(infer_img.shape[1])
                    infer_img=cv2.resize(infer_img,(int(infer_img.shape[1]*scale), int(infer_img.shape[0]*scale)), cv2.INTER_AREA)
                results = yolo_model.predict(infer_img, conf=conf_th, verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1,y1,x2,y2 = box.xyxy[0].tolist()
                        if infer_img is not view:
                            sx=view.shape[1]/float(infer_img.shape[1]); sy=view.shape[0]/float(infer_img.shape[0])
                            x1,x2 = x1*sx, x2*sx; y1,y2 = y1*sy, y2*sy
                        x1+=offset[0]; x2+=offset[0]; y1+=offset[1]; y2+=offset[1]
                        cls = int(box.cls[0].item()) if hasattr(box.cls[0],'item') else int(box.cls[0])
                        name = yolo_names.get(cls, str(cls)).lower() if yolo_names else str(cls)
                        conf = float(box.conf[0]) if hasattr(box.conf[0],'__float__') else float(box.conf[0])
                        # filtro polígono
                        cx,cy = (x1+x2)/2.0,(y1+y2)/2.0
                        if POLY and not point_in_poly(cx,cy,POLY): continue
                        # filtros allow/deny
                        if ALLOW and name not in ALLOW: continue
                        if DENY and name in DENY: continue
                        dets.append((x1,y1,x2,y2,name,conf))
            except Exception as e:
                put_tag(frame, f"YOLO err: {str(e)[:28]}", (10, 70), bg=(0,0,255))

        # ===== TRACKING + DWELL + TRIPWIRE =====
        tracks = tracker.update(dets)
        objs_dump=[]; alerts=[]
        for tid,(x1,y1,x2,y2) in tracks.items():
            # label/conf da melhor detecção (IoU)
            label, conf = "obj", 1.0; best_iou=0.0
            for (dx1,dy1,dx2,dy2,ln,cf) in dets:
                ix1,iy1=max(x1,dx1),max(y1,dy1); ix2,iy2=min(x2,dx2),min(y2,dy2)
                inter=max(0,ix2-ix1)*max(0,iy2-iy1); area_t=(x2-x1)*(y2-y1); area_d=(dx2-dx1)*(dy2-dy1)
                iou=inter/(area_t+area_d-inter+1e-6)
                if iou>best_iou: best_iou=iou; label,conf=ln,cf
            color = CLASS_COLORS[label]
            draw_fancy_box(frame, x1,y1,x2,y2, color, 2)

            # dwell
            first = tracker.first_seen.get(tid, now)
            dwell = now - first
            put_tag(frame, f"ID {tid} | {label} {conf:.2f} | {dwell:.1f}s", (int(x1), max(22,int(y1)-6)), bg=(0,0,0))
            log_event("object", tid, label, conf, x1,y1,x2,y2, extra=f"dwell={dwell:.2f}")
            objs_dump.append({"id":tid,"label":label,"conf":float(conf),"bbox":[float(x1),float(y1),float(x2),float(y2)],"dwell":dwell})

            # alerta por dwell
            if ARGS.dwell_alert>0 and dwell>=ARGS.dwell_alert:
                alerts.append(f"dwell:{tid}")
            # tripwire crossing
            prev = tracker.last_center.get(tid, ((x1+x2)/2.0,(y1+y2)/2.0))
            curr = ((x1+x2)/2.0,(y1+y2)/2.0)
            if TRIPWIRE and len(TRIPWIRE)==2:
                if seg_intersect(prev, curr, TRIPWIRE[0], TRIPWIRE[1]):
                    alerts.append(f"tripwire:{tid}")
                    log_event("line_cross", tid, label, conf, x1,y1,x2,y2, extra="tripwire")
            tracker.last_center[tid]=curr

            # alerta por classe
            if ALERT_CLASSES and label in ALERT_CLASSES:
                alerts.append(f"class:{label}")

        # ALERT handling
        if alerts:
            beep()
            if ARGS.screenshot_on_alert: screenshot(frame, "alert")

        # ===== FACES =====
        faces_dump=[]; hands_dump=[]
        if USE_FACES and face_det is not None:
            rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out=face_det.process(rgb)
            if out and out.detections:
                for det in out.detections:
                    box=det.location_data.relative_bounding_box
                    x1=int(box.xmin*w); y1=int(box.ymin*h); x2=int((box.xmin+box.width)*w); y2=int((box.ymin+box.height)*h)
                    x1,y1=max(0,x1),max(0,y1); x2,y2=min(w-1,x2),min(h-1,y2)
                    if POLY:
                        cx,cy=(x1+x2)/2.0,(y1+y2)/2.0
                        if not point_in_poly(cx,cy,POLY): continue
                    if FACE_BLUR and (x2>x1 and y2>y1):
                        face=frame[y1:y2,x1:x2]; 
                        if face.size>0: frame[y1:y2,x1:x2]=cv2.GaussianBlur(face,(31,31),0)
                    else:
                        draw_fancy_box(frame,x1,y1,x2,y2,(0,255,0),2)
                        put_tag(frame,"Face",(x1,max(22,y1-6)),bg=(0,128,0))
                        emo=""
                        if USE_EMOTION and DeepFace is not None and (x2>x1 and y2>y1):
                            emo=analyze_emotion(frame[y1:y2,x1:x2], _bbox_key(x1,y1,x2,y2))
                            if emo: put_tag(frame,f"Emoção: {emo}",(x1,y2+24),bg=(60,60,60))
                        faces_dump.append({"bbox":[x1,y1,x2,y2],"emotion":emo})
                    log_event("face", 0, "face", 1.0, x1,y1,x2,y2)

        # ===== MÃOS =====
        if USE_HANDS and hands is not None and not PAUSED:
            rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res=hands.process(rgb)
            if res.multi_hand_landmarks:
                try: handed=[h.classification[0].label for h in res.multi_handedness]
                except Exception: handed=["Right"]*len(res.multi_hand_landmarks)
                for i,lm in enumerate(res.multi_hand_landmarks):
                    mp_draw.draw_landmarks(frame,lm, mp_draw.HAND_CONNECTIONS)
                    dedos=fingers_state(lm.landmark,w,h, handed[i] if i<len(handed) else "Right")
                    name=gesture_name(dedos)
                    px=int(lm.landmark[8].x*w); py=int(lm.landmark[8].y*h)
                    put_tag(frame,name,(max(0,px-50), max(0,py-12)), bg=(40,40,40))
                    hands_dump.append({"bbox":[px-20,py-20,px+20,py+20],"gesture":name})
                    log_event("gesture",0,name,1.0, px-20,py-20,px+20,py+20)

        # ===== HUD / formas =====
        if SHOW_GRID: draw_grid(frame,w,h)
        draw_shapes(frame)
        put_tag(frame, f"FPS: {fps_smooth:.1f}", (10,30), bg=(0,0,0))
        status=f"[O:{'ON' if USE_OBJECTS else 'off'} F:{'ON' if USE_FACES else 'off'} E:{'ON' if USE_EMOTION else 'off'} H:{'ON' if USE_HANDS else 'off'} A-REC:{'ON' if ARGS.auto_rec else 'off'}] {ARGS.width}x{ARGS.height} | SRC {src}"
        put_tag(frame,status,(10,h-10),bg=(0,0,0))
        if SHOW_HELP: draw_help(frame,w,h)

        # ===== JSON dump =====
        dump_json(frame_id, w,h, objs_dump, faces_dump, hands_dump, {"alerts":alerts, "poly_points":POLY, "tripwire":TRIPWIRE})

        # ===== Auto-REC =====
        if ARGS.auto_rec:
            moving = (now - last_motion_ts) <= 0.3
            should_record = moving or (writer is not None and (now - last_motion_ts) < ARGS.rec_cooldown)
            if should_record and writer is None:
                fname=f"rec_{time.strftime('%Y%m%d_%H%M%S')}.mp4" if ARGS.rec_mp4 else f"rec_{time.strftime('%Y%m%d_%H%M%S')}.avi"
                writer=ensure_writer(os.path.join(ARGS.save_dir,fname), fps_smooth, (w,h))
                print(f"[AUTO-REC] Gravando: {fname}")
            if writer is not None and not should_record:
                writer.release(); writer=None; print("[AUTO-REC] Encerrado.")

        cv2.imshow("AI Vision", frame)
        if writer is not None: writer.write(frame)

        # ===== Teclado =====
        k=cv2.waitKey(1) & 0xFF
        if k==ord('q'): break
        elif k==ord(' '): PAUSED=not PAUSED
        elif k==ord('n'):
            PAUSED=True; ok,fr=cap.read(); 
            if ok: frame=cv2.flip(fr,1)
        elif k==ord('o'): USE_OBJECTS=not USE_OBJECTS
        elif k==ord('f'): USE_FACES=not USE_FACES
        elif k==ord('e'): USE_EMOTION=not USE_EMOTION
        elif k==ord('h'): USE_HANDS=not USE_HANDS
        elif k==ord('k'): SHOW_HELP=not SHOW_HELP
        elif k==ord('g'): SHOW_GRID=not SHOW_GRID
        elif k==ord('b'): FACE_BLUR=not FACE_BLUR
        elif k==ord('m'): MOTION_ONLY=not MOTION_ONLY
        elif k==ord('a'): ARGS.auto_rec=not ARGS.auto_rec
        elif k==ord('p'): screenshot(frame,"shot")
        elif k==ord('r'):
            if not ARGS.auto_rec:
                if writer is None:
                    fname=f"rec_{time.strftime('%Y%m%d_%H%M%S')}.mp4" if ARGS.rec_mp4 else f"rec_{time.strftime('%Y%m%d_%H%M%S')}.avi"
                    writer=ensure_writer(os.path.join(ARGS.save_dir,fname), fps_smooth, (w,h))
                    print(f"[REC] Gravando: {fname}")
                else:
                    writer.release(); writer=None; print("[REC] Finalizado.")
        elif k==ord('x'): POLY.clear()
        elif k==ord('z'): POLY_MODE=not POLY_MODE
        elif k==ord('t'): TW_MODE=not TW_MODE
        elif k==ord('c'):
            if ARGS.video is None:
                (cap,src2), cam_index = safe_reopen(cap, cam_index if cam_index is not None else -1, None)
                if not cap: print("[ERRO] Sem mais câmeras."); break
                src=src2; cap.set(cv2.CAP_PROP_FRAME_WIDTH,ARGS.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,ARGS.height)

    if writer: writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
