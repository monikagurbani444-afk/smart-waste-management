from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    "dataset/train",          # 👈 apne training folder ka path
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical"
)

print("Class indices mapping:", train_gen.class_indices)