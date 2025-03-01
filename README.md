# Emotion Detection Project

This project uses deep learning to perform **emotion detection** from images and real-time video.

---

## **1. Install Python 3.11**
Ensure Python **3.11** is installed.

- **Windows:** Download from [python.org](https://www.python.org/downloads/)
- **Mac:** Install via Homebrew:
  ```sh
  brew install python@3.11
  ```
- Verify installation:
  ```sh
  python --version
  ```

---

## **2. Download FER 2013 Dataset**
Download the **FER 2013 dataset** from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and store it in the `data/` folder.

---

## **3. Create a Virtual Environment**
Create a virtual environment named `emotion-detection`.

- **Windows:**
  ```sh
  python -m venv emotion-detection
  ```
- **Mac/Linux:**
  ```sh
  python3 -m venv emotion-detection
  ```

---

## **4. Install Dependencies**

### **Activate Virtual Environment:**
- **Windows:**
  ```sh
  emotion-detection\Scripts\activate
  ```
- **Mac/Linux:**
  ```sh
  source emotion-detection/bin/activate
  ```

### **Install Required Libraries:**
- Install dependencies from `requirements.txt`:
  ```sh
  pip install -r requirements.txt
  ```
- If any library is missing, install manually:
  ```sh
  pip install library_name
  ```

---

## **5. Create Images Folder**
Create an `images/` folder inside your project and add dummy images.

---

## **6. Modify the Model (Optional)**
To use a different model, update these scripts:
- `StaticDetection.py` → Handles image-based emotion detection.
- `RealTimeDetection.py` → Handles real-time emotion detection.

---

## **7. Modify Camera Source (If Needed)**
If your laptop **does not have an inbuilt camera**, update `RealTimeDetection.py`:

Change:
```python
cap = cv2.VideoCapture(0)
```
To:
```python
cap = cv2.VideoCapture(1)
```

---

## **8. Run the Scripts**
- **Run Static Image Detection:**
  ```sh
  python StaticDetection.py
  ```
- **Run Real-Time Detection:**
  ```sh
  python RealTimeDetection.py
  ```

---

## **9. Deactivate Virtual Environment**
After execution, deactivate the virtual environment:
```sh
deactivate
```

---

### ✅ **Your Emotion Detection System is Ready!** 🎉

