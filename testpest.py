import numpy as np
from PIL import Image
import tensorflow as tf

print("🔥 Loading model...")
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
print("✅ Ready!")

# Your pests
LIGHTS = {0: "380nm UV-Fall Armyworm", 1: "520nm Green-Hopper", 2: "OFF-Healthy", 3: "480nm Blue-Locust", 4: "Trap-Mouse"}

# Test photo (CHANGE YOUR FILENAME)
img = Image.open("test.jpg").resize((224,224))  # ← YOUR PHOTO
data = np.array(img, dtype=np.uint8)
data = np.expand_dims(data, axis=0)

interpreter.set_tensor(interpreter.get_input_details()[0]["index"], data)
interpreter.invoke()
result = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]

pest = np.argmax(result)
print(f"\n PEST #{pest}: {LIGHTS[pest]} ({result[pest]*100:.1f}%)")