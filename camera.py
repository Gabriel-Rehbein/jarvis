import cv2, threading, time

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.lock = threading.Lock()
        self.running = True
        self.detect_faces = True
        self.detect_objects = False
        self.last_frame = None

    def __del__(self):
        self.cap.release()

    def toggle(self, feature):
        if feature == "faces":
            self.detect_faces = not self.detect_faces
            return self.detect_faces
        elif feature == "objects":
            self.detect_objects = not self.detect_objects
            return self.detect_objects
        return None

    def status(self):
        return {
            "faces": self.detect_faces,
            "objects": self.detect_objects,
        }

    def generate(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            if self.detect_faces:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
