import cv2
import tensorflow as tf
from collections import Counter

# Load the saved model
model = tf.saved_model.load("signs")

# Set up video capture
cap = cv2.VideoCapture(0)
# Set up object tally
tally = Counter()


while True:
    # Read frame from video capture
    ret, frame = cap.read()

    # Preprocess frame for model input
    input_frame = cv2.resize(frame, (224, 224))
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_frame = input_frame.astype('float32')
    input_frame = input_frame / 255.0
    
    input_frame = input_frame.reshape(1, 224, 224, 3)

    predictions = model(input_frame)

    # Compute index of maximum value along last axis
    predicted_class = tf.argmax(predictions, axis=-1)[0]
    key=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
    mazwi=''
    # Convert predicted_class to integer
    predicted_class = int(predicted_class)
    mazwi +=key[predicted_class]
    confidence = predictions[0][predicted_class]
    # Update tally
    tally[predicted_class] += 1
    # Display frame and tally
    cv2.imshow('Frank', frame)
    print(mazwi, confidence)

    # Check for exit key
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()