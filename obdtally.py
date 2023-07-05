import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2

# Load the saved model
model = tf.keras.models.load_model('saved_model')

def object_detection(image, min_conf=0.9):
    # Initialize variables used for the object detection procedure
    PYR_SCALE = 1.5
    WIN_STEP = 16
    ROI_SIZE = (200, 150)
    INPUT_SIZE = (224, 224)
    
    # Initialize the image pyramid
    pyramid = image_pyramid(image, scale=PYR_SCALE, minSize=ROI_SIZE)
    
    # Initialize two lists, one to hold the ROIs generated from the image pyramid and sliding window,
    # and another list used to store the (x, y)-coordinates of where the ROI was in the original image
    rois = []
    locs = []
    
    # Loop over the image pyramid
    for image in pyramid:
        # Determine the scale factor between the *original* image dimensions and the *current* layer of the pyramid
        scale = W / float(image.shape[1])
        
        # For each layer of the image pyramid, loop over the sliding window locations
        for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
            # Scale the (x, y)-coordinates of the ROI with respect to the *original* image dimensions
            x = int(x * scale)
            y = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)
            
            # Take the ROI and preprocess it so we can later classify the region using Keras/TensorFlow
            roi = cv2.resize(roiOrig, INPUT_SIZE)
            roi = img_to_array(roi)
            roi = preprocess_input(roi)
            
            # Update our list of ROIs and associated coordinates
            rois.append(roi)
            locs.append((x, y, x + w, y + h))
    
    # Convert the ROIs to a NumPy array
    rois = np.array(rois, dtype="float32")
    
    # Classify each of the proposal ROIs using ResNet and then show how long the classifications took
    preds = model.predict(rois)
    
    # Decode the predictions and initialize a dictionary which maps class labels (keys) to any ROIs associated with that label (values)
    preds = imagenet_utils.decode_predictions(preds, top=1)
    labels = {}
    
    # Loop over the predictions
    for (i, p) in enumerate(preds):
        # Grab the prediction information for the current ROI
        (imagenetID, label, prob) = p[0]
        
        # Filter out weak detections by ensuring the predicted probability is greater than the minimum probability
        if prob >= min_conf:
            # Grab the bounding box associated with the prediction and convert the coordinates
            box = locs[i]
            
            # Grab the list of predictions for the label and add the bounding box + probability to the list
            L = labels.get(label, [])
            L.append((box, prob))
            labels[label] = L
    
    # Loop over the labels for each of detected objects in the image
    for label in labels.keys():
        # Clone the original image so that we can draw on it.
        print("[INFO] showing results for '{}'".format(label))
        clone = orig.copy()
        
        # Extract the bounding boxes and associated prediction probabilities,
        # then apply non-maxima suppression.
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, proba)
        
       # Loop over all bounding boxes that were kept after applying non-maxima suppression
objects = {}
for (startX, startY, endX, endY) in boxes:
    # Draw the bounding box on the image
    cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
    # Get the label and probability associated with the bounding box
    label = labels[label][0][1]
    prob = labels[label][0][1]
    
    # Add the label and probability to the objects dictionary
    if label not in objects:
        objects[label] = []
    objects[label].append(prob)

# Print out the tally of objects found
for label in objects:
    print("Found {} {}(s) with an average confidence of {:.2f}".format(len(objects[label]), label, sum(objects[label])/len(objects[label])))