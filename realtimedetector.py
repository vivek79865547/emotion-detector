import cv2
from keras.models import model_from_json
import numpy as np

# Load the model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haarcascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Labels
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}

# Feature extraction
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature / 255.0

# Phobia Detection Logic (only emotion-based)
def detect_phobia(emotion):
    fear_emotions = ['fear']
    if emotion.lower() in fear_emotions:
        return True
    else:
        return False

# Webcam
webcam = cv2.VideoCapture(0)

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            # Detect phobia based only on emotion
            phobia_detected = detect_phobia(prediction_label)

            # Display
            if phobia_detected:
                text = f"PHOBIA DETECTED! ({prediction_label})"
                color = (0, 0, 255)  # Red
            else:
                text = f"Emotion: {prediction_label}"
                color = (0, 255, 0)  # Green

            cv2.putText(im, text, (p-10, q-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Phobia Detector", im)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()

