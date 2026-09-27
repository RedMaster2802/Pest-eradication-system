import tensorflow as tf
import numpy as np
from PIL import Image

# Speed of light
c = 3 * 10**8

# Load model
model = tf.keras.models.load_model('Model.h5')

classes = ['Fall Armyworm', 'Brown Plant Hopper', 'Healthy']

wavelength_map = {
    'Fall Armyworm': 365e-9,
    'Brown Plant Hopper': 450e-9,
    'Healthy': None
}

# Load image
img = Image.open('test.jpg').convert('RGB')
img = img.resize((224, 224))

# Preprocess
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
index = np.argmax(prediction)
pest = classes[index]

print("Detected:", pest)

if wavelength_map[pest]:
    wavelength = wavelength_map[pest]
    frequency = c / wavelength

    print(f"Wavelength: {wavelength*1e9:.0f} nm")
    print(f"Frequency: {frequency:.2e} Hz")
else:
    print("No pest detected")