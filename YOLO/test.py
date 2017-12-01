import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import scipy.io
import scipy.mise

from main import yolo_eval
from keras import backend as K
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

from keras.layers import Input, Lambda, Conv2D
from keras.model import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes

sess = K.get_session()

# Defining classes, anchors and image shape

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)

# Loading a pretrained model

yolo_model = load_model("model_data/yolo.h5")

yolo_model.summary()

# Convert output of the model to usable bounding box tensors

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

# Filtering boxes

scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

# Run the graph on an image

def predict(sess, image_file):
    image, image_data = preprocess_image("image/" + image_file, model_image_size = (608, 608))
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    colors = generate_colors(class_names)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    image.save(os.path.join("out", image_file), quality=90)
    output_image = scipy.misc.imread(os.path.join("out", image_file))

    imshow(output_image)
    return out_scores, out_boxes, out_classes

out_scores, out_boxes, out_classes = predict(sess, "test.jpg")