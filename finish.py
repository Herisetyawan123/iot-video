from flask import Flask, Response
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
import time
from threading import Thread

# Load model TensorFlow Lite
tflite_model_path = "model_deteksi_penyakit_ayam.tflite"  # Path ke model TFLite Anda
interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Mendapatkan detail tensor input dan output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label kategori
categories = ["kolera", "sehat", "snot"]  # Ganti sesuai dengan kategori Anda


# Fungsi untuk memproses ROI (Region of Interest)
def process_roi(roi):
    input_size = (150, 150)  # Sesuaikan dengan ukuran input model Anda
    resized_roi = cv2.resize(roi, input_size)
    normalized_roi = resized_roi / 255.0
    image_array = np.expand_dims(normalized_roi, axis=0).astype(np.float32)

    # Prediksi dengan model TFLite
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    class_index = np.argmax(predictions)
    confidence = predictions[class_index]

    return categories[class_index], confidence


# Fungsi untuk mendeteksi kepala ayam berdasarkan warna merah
def detect_head(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rentang warna merah
    lower_red1 = np.array([0, 70, 50])  # Rentang pertama
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])  # Rentang kedua
    upper_red2 = np.array([180, 255, 255])

    # Masking merah
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Deteksi kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:  # Threshold area
            return cv2.boundingRect(largest_contour)
    return None


# Set up GPIO for servo motor
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)
servo1 = GPIO.PWM(11, 50)  # Pin 11, 50Hz pulse
servo1.start(0)


def move_servo(direction):
    if direction == "forward":
        print("Moving forward")
        # duty = 2
        servo1.ChangeDutyCycle(12)
        # while duty <= 12:
        #     time.sleep(1)
        #     duty = duty + 1
    elif direction == "backward":
        print("Moving backward")
        # duty = 12
        servo1.ChangeDutyCycle(2)
        # while duty >= 2:
        #     time.sleep(1)
        #     duty = duty - 1
    else:
        print("Stopping")
        servo1.ChangeDutyCycle(0)
        time.sleep(0.5)


# Flask app
app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(0)  # Ubah dengan sumber kamera Anda
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Deteksi kepala ayam
        bounding_box = detect_head(frame)
        if bounding_box:
            x, y, w, h = bounding_box
            chicken_head = frame[y : y + h, x : x + w]
            label, confidence = process_roi(chicken_head)
            confidence_percentage = f"{confidence * 100:.2f}%"

            # Gambar kotak hijau di sekitar kepala ayam
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} ({confidence_percentage})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Ayam tidak terdeteksi",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Encode frame ke JPEG
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        # Kirim frame sebagai HTTP response
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/move_servo/<direction>")
def move_servo_endpoint(direction):
    # Menjalankan gerakan servo di thread terpisah
    thread = Thread(target=move_servo, args=(direction,))
    thread.start()
    return f"Servo moving {direction}"


@app.route("/")
def index():
    return """
    <html>
        <head>
            <title>Deteksi Ayam</title>
        </head>
        <body>
            <h1>Deteksi Ayam</h1>
            <img src="/video_feed" width="640" height="480">
            <br>
            <button onclick="window.location.href='/move_servo/forward'">Move Forward</button>
            <button onclick="window.location.href='/move_servo/backward'">Move Backward</button>
        </body>
    </html>
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

