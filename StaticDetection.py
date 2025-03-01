import cv2
import numpy as np
from keras.models import model_from_json, load_model

emotion_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad',6:'Surprise'}

# load json and create model
json_file = open('model/model_20.json','r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into model
emotion_model.load_weights('model/model_20.weights.h5')
print('Loaded model from disk')

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

frame = cv2.imread("images/sad.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray, 1.3, )

# Process each detected face
for (x, y, w, h) in faces:
    # Extract the region of interest (ROI) containing the face
    roi_gray = gray[y:y + h, x:x + w]

    # Resize the ROI to match MobileNetV2 input size
    roi_resized = cv2.resize(roi_gray, (48, 48))

    # Add two extra channels filled with zeros
    roi_color = np.expand_dims(roi_resized, axis=-1)  # Expand to add a single channel
    roi_color = np.repeat(roi_color, 3, axis=-1)  # Repeat 3 times to create 3 channels

    # Normalize the ROI
    roi_color = roi_color / 255.0

    # Reshape the ROI to match the expected input shape
    roi_color = roi_color.reshape(1, 48, 48, 3)

    # Predict the emotion label
    predicted_emotion = emotion_model.predict(roi_color)

    # Get the index of the predicted emotion label
    predicted_emotion_index = int(predicted_emotion.argmax())

    # Get the corresponding emotion label from the emotion dictionary
    emotion_label = emotion_dict[predicted_emotion_index]

    # Draw a rectangle around the detected face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the predicted emotion label above the detected face
    cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Display the processed image with detected faces and predicted emotions
cv2.imshow('Emotion Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()