import cv2
import numpy as np
from keras.models import model_from_json
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


emotion_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad',6:'Surprise'}

# Load the model architecture and weights
json_file = open('model/model_20.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
# Load architecture
# with open('model/model_20.json', 'r') as json_file:
#     model_json = json_file.read()
# emotion_model = model_from_json(model_json)

# Load weights
emotion_model.load_weights('model/model_20.weights.h5')
# emotion_model.load_weights('model/model_20.weights.h5')
print('Loaded model from disk')

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))  # Resize frame if needed

    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using cascade classifier
    num_faces = faceDetect.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in num_faces:
        # Draw rectangle around detected faces
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)

        # Extract the region of interest (ROI) for the face
        roi_gray_frame = gray_frame[y:y+h, x:x+w]

        # Preprocess the ROI for model prediction:
        # 1. Resize to match MobileNetV2 input size
        roi_resized = cv2.resize(roi_gray_frame, (48, 48))

        # 2. Add two extra channels filled with zeros
        roi_color = np.expand_dims(roi_resized, axis=-1)  # Expand to add a single channel
        roi_color = np.repeat(roi_color, 3, axis=-1)  # Repeat 3 times to create 3 channels

        # 3. Normalize (optional, might have been done during training)
        roi_color = roi_color / 255.0

        # 4. Reshape to match expected input shape
        cropped_img = roi_color.reshape(1, 48, 48, 3)

        # Make prediction using the model
        emotion_prediction = emotion_model.predict(cropped_img)
        max_index = int(np.argmax(emotion_prediction))

        # Display predicted emotion label on the frame
        cv2.putText(frame, emotion_dict[max_index], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)

    # Display the frame with detected faces and emotions
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
