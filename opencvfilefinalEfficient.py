import cv2
import tensorflow as tf
from collections import Counter

# Load the saved model
model = tf.saved_model.load("EffecientNetB3-Tomato-Disease-Detection-99.76")

# Set up video capture
cap = cv2.VideoCapture(0)
# Set up object tally
tally = Counter()


while True:
    # Read frame from video capture
    ret, frame = cap.read()

    # Preprocess frame for model input
    input_frame = frame
    input_frame = cv2.resize(frame, (224, 224))
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_frame = input_frame.astype('float32')
    #input_frame = input_frame / 255.0
   
    input_frame = input_frame.reshape(1, 224, 224, 3)

    prediction = model(input_frame)

    # Compute index of maximum value along last axis
    predicted_class = tf.argmax(prediction, axis=-1)[0]

   # Convert predicted_class to integer
    predicted_class = int(predicted_class)

    # Update tally
    tally[predicted_class] += 1
    # Display frame and tally
    cv2.imshow('Frame', frame)
    print(tally)

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()