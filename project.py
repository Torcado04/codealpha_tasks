import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

MODEL_PATH = "digit_model.keras"



# LOAD OR TRAIN MODEL

def get_model():
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        return keras.models.load_model(MODEL_PATH)

    print("Training model...")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = keras.models.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=5)

    model.save(MODEL_PATH)
    return model



# PREPROCESS DIGIT

def preprocess_digit(img):
    coords = np.column_stack(np.where(img > 0))

    # Handle empty image safely
    if coords.size == 0:
        return np.zeros((1, 28, 28, 1))

    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]

    h, w = img.shape

    if h > w:
        new_h, new_w = 20, int(w * 20 / h)
    else:
        new_w, new_h = 20, int(h * 20 / w)

    img = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2

    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = img

    canvas = canvas / 255.0
    canvas = canvas.reshape(1, 28, 28, 1)

    return canvas



# PREDICT FROM IMAGE

def predict_image(path, model):
    img = cv2.imread(path)

    if img is None:
        print("Image not found")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Better threshold (auto)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    # Clean noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # More realistic filter
        if area < 300:
            continue

        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])

    results = []

    for (x, y, w, h) in boxes:
        digit = thresh[y:y+h, x:x+w]

        processed = preprocess_digit(digit)

        pred = model.predict(processed, verbose=0)
        digit_class = np.argmax(pred)

        results.append(str(digit_class))

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, str(digit_class), (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    final = "".join(results)

    print("Final Prediction:", final)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Digits: " + final)
    plt.axis('off')
    plt.show()



# TEST WITH MNIST

def test_with_sample(model):
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_test = x_test / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Random test instead of fixed index
    index = np.random.randint(0, len(x_test))

    img = x_test[index]
    true_label = y_test[index]

    pred = model.predict(img.reshape(1, 28, 28, 1), verbose=0)
    predicted_label = np.argmax(pred)

    print("True Label:", true_label)
    print("Predicted Label:", predicted_label)

    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()



# MAIN

if __name__ == "__main__":
    model = get_model()

    # Test with dataset
    # test_with_sample(model)

    # Uncomment to test with real image
    predict_image("digit.png", model)