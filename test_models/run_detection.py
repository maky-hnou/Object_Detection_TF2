import cv2
import numpy as np

import tensorflow as tf


def create_category_index(categories):
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index


class RunDetection:
    def __init__(self, path_to_model, path_to_labelmap,
                 class_id=None, threshold=0.5):
        self.Threshold = threshold
        categories = [{"name": "ped", "id": 1}]
        self.category_index = create_category_index(categories)

        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(path_to_model)

    def detect_from_image(self, image):
        height, width = image.shape[:2]
        # Expand dimensions since the model expects images to have
        #  shape: [1, None, None, 3]
        input_tensor = np.expand_dims(image, 0)
        detections = self.model(input_tensor)

        bboxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()
        det_boxes = self.ExtractBBoxes(bboxes, classes, scores, width, height)

        return det_boxes

    def extract_bboxes(self, bboxes, classes, scores, width, height):
        bbox = []
        for idx in range(len(bboxes)):
            if scores[idx] >= self.Threshold:
                y_min = int(bboxes[idx][0] * height)
                x_min = int(bboxes[idx][1] * width)
                y_max = int(bboxes[idx][2] * height)
                x_max = int(bboxes[idx][3] * width)
                class_label = self.category_index[int(classes[idx])]['name']
                bbox.append([x_min, y_min, x_max,
                             y_max, class_label, float(scores[idx])])
        return bbox

    def display_output_image(self, image, boxes_list, det_time=None):
        if not boxes_list:
            return image
        for idx in range(len(boxes_list)):
            x_min = boxes_list[idx][0]
            y_min = boxes_list[idx][1]
            x_max = boxes_list[idx][2]
            y_max = boxes_list[idx][3]
            cls = str(boxes_list[idx][4])
            score = str(np.round(boxes_list[idx][-1], 2))

            text = cls + ": " + score
            cv2.rectangle(image, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 1)
            cv2.rectangle(image, (x_min, y_min - 20),
                          (x_min, y_min), (255, 255, 255), -1)
            cv2.putText(image, text, (x_min + 5, y_min - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if det_time is not None:
            fps = round(1000. / det_time, 1)
            fps_txt = str(fps) + " FPS"
            cv2.putText(image, fps_txt, (25, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 2)

        return image
