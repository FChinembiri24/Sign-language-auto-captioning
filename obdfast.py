import cv2
import numpy as np
import tensorflow as tf
from imutils.object_detection import non_max_suppression

# Load TensorFlow SavedModel
model = tf.saved_model.load('EffecientNetB3-Tomato-Disease-Detection-99.76')

def sliding_window(image, win_size, win_step):
    for y in range(0, image.shape[0] - win_size[1], win_step):
        for x in range(0, image.shape[1] - win_size[0], win_step):
            yield (x, y, image[y:y + win_size[1], x:x + win_size[0]])

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
        boxes = []
        for (x, y, roi) in sliding_window(layer, win_size, win_step):
            # Step #3a: For each stop of the sliding window, extract the ROI
            roi = cv2.resize(roi, (224, 224))
            roi = tf.keras.applications.efficientnet.preprocess_input(roi)
            boxes.append((x * scale, y * scale, (x + win_size[0]) * scale, (y + win_size[1]) * scale))
            rois.append(roi)
        rois = np.array(rois)

        # Step #3b: Take the ROI and pass it through our CNN originally trained for image classification
        preds = model(rois)
        labels = tf.keras.applications.efficientnet.decode_predictions(preds)

        # Step #3c: Examine the probability of the top class label of the CNN,
        # and if meets a minimum confidence, record (1) the class label and (2) the location of the sliding window
        for i in range(len(labels)):
            if labels[i][0][2] > min_conf:
                box = boxes[i]
                rois.append((box, labels[i][0][2]))
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
   
    # Preprocess frame for model input
    input_frame = cv2.resize(frame, (224, 224))
    #input_frame = input_frame / 255.0
    input_frame = input_frame.astype('float32')
    input_frame = input_frame.reshape(1, 224, 224, 3)
   
    # Perform object detection on frame
    boxes = object_detection(input_frame)
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