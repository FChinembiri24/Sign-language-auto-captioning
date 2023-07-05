import cv2
import numpy as np
import tensorflow as tf
from imutils.object_detection import non_max_suppression

# Load TensorFlow SavedModel
model = tf.saved_model.load('path/to/saved_model')

def object_detection(image, win_size=(64, 64), win_step=4, pyramid_scale=1.5, min_conf=0.5):
    # Step #1: Input an image
    (H, W) = image.shape[:2]

    # Step #2: Construct an image pyramid
    pyramid = []
    scale = 1
    while W // scale >= win_size[1] and H // scale >= win_size[0]:
        pyramid.append(cv2.resize(image, (W // scale, H // scale)))
        scale *= pyramid_scale

    # Step #3: For each scale of the image pyramid, run a sliding window
    rois = []
    locs = []
    for layer in pyramid:
        scale = W / float(layer.shape[1])
        for y in range(0, layer.shape[0] - win_size[1], win_step):
            for x in range(0, layer.shape[1] - win_size[0], win_step):
                # Step #3a: For each stop of the sliding window, extract the ROI
                roi = layer[y:y + win_size[1], x:x + win_size[0]]

                # Step #3b: Take the ROI and pass it through our CNN originally trained for image classification
                roi = cv2.resize(roi, (224, 224))
                roi = tf.keras.applications.efficientnet.preprocess_input(roi)
                preds = model(np.expand_dims(roi, axis=0))
                label = tf.keras.applications.efficientnet.decode_predictions(preds)[0][0]

                # Step #3c: Examine the probability of the top class label of the CNN,
                # and if meets a minimum confidence, record (1) the class label and (2) the location of the sliding window
                if label[2] > min_conf:
                    box = (int(x * scale), int(y * scale), int((x + win_size[0]) * scale), int((y + win_size[1]) * scale))
                    rois.append((box, label[2]))
                    locs.append(box)

    # Step #4: Apply class-wise non-maxima suppression to the bounding boxes
    boxes = np.array([p[0] for p in rois])
    proba = np.array([p[1] for p in rois])
    boxes = non_max_suppression(boxes, proba)

    # Step #5: Return results to calling function
    return boxes

# Set up video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Perform object detection on frame
    boxes = object_detection(frame)

    # Draw bounding boxes on frame
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Output', frame)

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()