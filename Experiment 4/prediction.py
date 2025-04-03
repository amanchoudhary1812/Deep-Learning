from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import pyttsx3

model = load_model("image_classifier.h5")
obj=pyttsx3.init()

def speak(text):
    
    obj.say(text)
    obj.runAndWait()

img = image.load_img("pre/1.jpg", target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
class_index = np.argmax(prediction)
if(class_index==0):
    print("Predicted class: Car")
    speak("It is a car")
elif(class_index==1):
    print("Predicted class: flower")
    speak("It is a flower")

elif(class_index==2):
    print("Predicted class: plane")
    speak("It is a plane")

else:
    print("Predicted class: unknown")


# print("Predicted class:", class_index)
