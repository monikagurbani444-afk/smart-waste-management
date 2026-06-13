import os, random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix

# Load trained model
model = load_model("model/waste_model.h5")

# Paths
test_dir = "dataset/test"
categories = ["Organic", "Plastic", "Metal", "E-waste"]

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# 🔹 Predict on one random image from each category
for cat in categories:
    cat_path = os.path.join(test_dir, cat)
    random_img = random.choice(os.listdir(cat_path))
    img_path = os.path.join(cat_path, random_img)

    img = load_img(img_path, target_size=(128,128))
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = categories[np.argmax(prediction)]

    print(f"Image tested: {img_path}")
    print(f"Predicted class: {predicted_class}\n")

# 🔹 Evaluate on full test set
loss, acc = model.evaluate(test_generator)
print(f"Overall Test Accuracy: {acc:.4f}")

# 🔹 Confusion matrix & classification report
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=categories))