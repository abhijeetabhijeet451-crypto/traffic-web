import os
import cv2
import tempfile
from flask import Flask, request, render_template_string, send_file
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolov8n.pt")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Smart Traffic Controller</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; background: #0e1117; color: white; margin: 0; padding: 20px; text-align: center; }
        .box { background: #262730; padding: 20px; border-radius: 10px; max-width: 800px; margin: auto; }
        input[type=file] { margin: 15px 0; color: white; }
        input[type=submit] { background: #ff4b4b; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .alert { padding: 15px; border-radius: 8px; margin-top: 15px; font-size: 18px; font-weight: bold; }
        video { width: 100%; border-radius: 8px; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="box">
        <h1>🚦 AI Smart Traffic Controller</h1>
        <p>Vehicle Density Optimization & Emergency Vehicle Detection</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="video" accept="video/*" required><br>
            <input type="submit" value="Upload & Analyze Traffic">
        </form>

        {% if status %}
            <div class="alert" style="background-color: {{ color }}; color: black;">
                {{ status }}
            </div>
            <p>Allocated Signal Timer: <b>{{ timer }}s</b></p>
            <video controls autoplay loop src="/get-video/{{ video_name }}"></video>
        {% endif %}
    </div>
</body>
</html>
"""

PROCESSED_FOLDER = tempfile.gettempdir()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("video")
        if file:
            in_path = os.path.join(PROCESSED_FOLDER, "input.mp4")
            out_name = "processed_traffic.mp4"
            out_path = os.path.join(PROCESSED_FOLDER, out_name)
            file.save(in_path)

            cap = cv2.VideoCapture(in_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360

            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            vehicle_count = 0
            emergency_detected = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False)
                boxes = results[0].boxes
                curr = 0
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls in [2, 3, 5, 7]:
                        curr += 1
                    if cls == 7:
                        emergency_detected = True

                if curr > vehicle_count:
                    vehicle_count = curr

                annotated = results[0].plot()
                out.write(annotated)

            cap.release()
            out.release()

            if emergency_detected:
                status = "🟢 GREEN OVERRIDE: Emergency Vehicle Detected!"
                color = "#28a745"
                timer = 60
            elif vehicle_count > 4:
                status = f"🟢 HIGH DENSITY: {vehicle_count} Vehicles Detected"
                color = "#28a745"
                timer = min(20 + (vehicle_count * 3), 60)
            else:
                status = "🔴 RED: Standard Traffic"
                color = "#dc3545"
                timer = 15

            return render_template_string(HTML_TEMPLATE, status=status, color=color, timer=timer, video_name=out_name)

    return render_template_string(HTML_TEMPLATE, status=None)

@app.route("/get-video/<filename>")
def get_video(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), mimetype="video/mp4")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
