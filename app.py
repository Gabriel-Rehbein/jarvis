from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
import os

app = Flask(__name__)
camera = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(camera.generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle', methods=['POST'])
def toggle_feature():
    data = request.json
    feature = data.get("feature")
    state = camera.toggle(feature)
    return jsonify({"feature": feature, "state": state})

@app.route('/status')
def status():
    return jsonify(camera.status())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
