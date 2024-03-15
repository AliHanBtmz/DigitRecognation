import math
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from scipy import ndimage

# Capture video from the default camera
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Define the input shape for the model
input_shape = (28, 28, 1)

# Load the pre-trained model
model = load_model('CNNwithMNİST.h5')


def getBestShift(img):
    # Calculate the best shift for the image
    cy, cx = ndimage.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty


def shift(img, sx, sy):
    # Shift the image
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def prediction(gray, model):
    # Preprocess the image and make prediction
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Remove any empty rows or columns from the edges
    while np.sum(gray[0]) == 0:
        gray = gray[1:]
    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)
    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]
    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)

    # Resize the image to 28x28 maintaining aspect ratio
    rows, cols = gray.shape
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        gray = cv2.resize(gray, (cols, rows))

    # Pad the image to 28x28
    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

    # Apply the best shift
    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted

    # Flatten and normalize the image
    flatten = gray.flatten() / 255.0
    flatten = flatten.reshape(1, 28, 28, 1)

    # Make prediction using the model
    outputs = model.predict(flatten)
    predicted = np.argmax(outputs, axis=1)
    return predicted


while True:
    # Capture frame-by-frame
    _, frame = cap.read()
    frame_copy = frame.copy()

    # Define bounding box
    bbox_size = (60, 60)
    bbox = [(int(WIDTH // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
            (int(WIDTH // 2 + bbox_size[0] // 2), int(HEIGHT // 2 + bbox_size[1] // 2))]

    # Crop the frame
    img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

    # Convert cropped image to grayscale and preprocess for prediction
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(255 - img_gray, (28, 28))

    # Get prediction for the cropped image
    result = prediction(img_gray, model)
    img_gray = cv2.resize(img_gray, (400, 400))
    cv2.imshow("cropped", img_gray)

    # Draw prediction result on the frame
    result = prediction(img_gray, model)
    cv2.putText(frame_copy, f"Prediction: {result}", (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (225, 0, 255), 2, cv2.LINE_AA)

    # Draw bounding box on the frame
    cv2.rectangle(frame_copy, bbox[0], bbox[1], (0, 255, 0), 3)

    # Display the resulting frame
    cv2.imshow("input", frame_copy)

    # Exit the loop when ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()