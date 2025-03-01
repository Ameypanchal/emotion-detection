import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from keras.models import model_from_json

# Define emotion dictionary
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load the model architecture and weights
json_file = open('model/model_20.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights('model/model_20.weights.h5')
print('Loaded model from disk')

# Load face cascade classifier
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create main Tkinter GUI window
root = tk.Tk()
root.title("Emotion Detection")

# Create a label to display detected emotions
emotion_label = tk.Label(root, font=('Helvetica', 20), pady=20)
emotion_label.pack()

# Create a canvas to display the webcam feed
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Initialize video capture
cap = cv2.VideoCapture(1)

# Function to detect emotions and update the GUI
def detect_emotions():
    global cap
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))  # Resize frame

    if ret:
        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using cascade classifier
        num_faces = faceDetect.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in num_faces:
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
            emotion_label.config(text=f"Emotion: {emotion_dict[max_index]}")

            # Display probabilities in the secondary window
            prob_window.update_probabilities(emotion_prediction)

        # Convert frame to RGB format for display in Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        # Update canvas with the new frame
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.img = img

        # Schedule the next update
        root.after(10, detect_emotions)

# Function to create secondary window for displaying probabilities
def create_prob_window():
    prob_window = ProbabilityWindow(root)
    return prob_window

# Class to create secondary window for displaying probabilities
class ProbabilityWindow:
    def __init__(self, master):
        self.master = master
        self.window = tk.Toplevel(master)
        self.window.title("Emotion Probabilities")
        self.labels = []

        # Create labels for displaying probabilities
        for i in range(len(emotion_dict)):
            label = tk.Label(self.window, text=f"{emotion_dict[i]}: 0.00%", font=('Helvetica', 12))
            label.pack()
            self.labels.append(label)

    # Function to update probability labels
    def update_probabilities(self, probabilities):
        for i, probability in enumerate(probabilities[0]):
            self.labels[i].config(text=f"{emotion_dict[i]}: {probability * 100:.2f}%")

# Start the secondary window for displaying probabilities
prob_window = create_prob_window()

# Start emotion detection
detect_emotions()

# Start the main GUI loop
root.mainloop()
