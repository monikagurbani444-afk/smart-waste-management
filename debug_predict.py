from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# ✅ Load model
model = load_model("model/waste_model.h5")

# ✅ Categories order fixed
categories = ["E-waste", "metal", "organic", "plastic"]

def test_prediction(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = categories[np.argmax(prediction)]
    confidence = np.max(prediction)

    print("🔹 Raw prediction array:", prediction)
    print("🔹 Predicted class:", predicted_class)
    print("🔹 Confidence:", round(confidence * 100, 2), "%")

# ✅ Example usage
test_prediction("dataset/test/organic/O_W (58).jpg")