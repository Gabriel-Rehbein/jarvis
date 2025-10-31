from flask import Flask, render_template, Response, jsonify, request, make_response
from camera import VideoCamera
import time

app = Flask(__name__)
cam = VideoCamera()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    # Stream MJPEG com cabeçalhos anti-cache
    def gen():
        for chunk in cam.mjpeg_generator():
            yield chunk
    resp = Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route("/snapshot")
def snapshot():
    jpg = cam.get_jpeg()
    if jpg is None:
        return "no frame", 503
    r = make_response(jpg)
    r.headers["Content-Type"] = "image/jpeg"
    r.headers["Cache-Control"] = "no-cache"
    return r

@app.route("/health")
def health():
    ok, msg = cam.health()
    return jsonify({"ok": ok, "msg": msg})

@app.route("/open_cam")
def open_cam():
    idx = request.args.get("index", default=None, type=int)
    ok, msg = cam.open_index(idx)
    return jsonify({"ok": ok, "msg": msg})

# ===== UI/flags =====
@app.route("/status")
def status():
    return jsonify(cam.status())

@app.route("/action", methods=["POST"])
def action():
    data = request.get_json(force=True, silent=True) or {}
    cmd = data.get("cmd")
    ok, payload = cam.action(cmd)
    return (jsonify({"ok": ok, "data": payload}), 200 if ok else 400)

@app.route("/set_conf", methods=["POST"])
def set_conf():
    data = request.get_json(force=True, silent=True) or {}
    conf = float(data.get("conf", 0.5))
    conf = max(0.01, min(1.0, conf))
    cam.set_conf(conf)
    return jsonify({"ok": True, "conf": conf})

@app.route("/set_roi", methods=["POST"])
def set_roi():
    data = request.get_json(force=True, silent=True) or {}
    cam.set_roi_norm(float(data.get("x1",0.0)), float(data.get("y1",0.0)),
                     float(data.get("x2",1.0)), float(data.get("y2",1.0)))
    return jsonify({"ok": True})

@app.route("/clear_roi", methods=["POST"])
def clear_roi():
    cam.clear_roi()
    return jsonify({"ok": True})

if __name__ == "__main__":
    # IMPORTANTE no Windows: evite o reloader (abre a câmera 2x).
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)
