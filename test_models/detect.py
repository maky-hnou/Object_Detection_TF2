import os
import time

import cv2


def detect_from_images(detector, images_dir, save_output=False, output_dir='output/'):
    for file in os.scandir(images_dir):
        if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_dir, file.name)
            print(image_path)
            img = cv2.imread(image_path)
            det_boxes = detector.DetectFromImage(img)
            img = detector.DisplayDetections(img, det_boxes)

            cv2.imshow('TF2 Detection', img)
            cv2.waitKey(0)

            if save_output:
                img_out = os.path.join(output_dir, file.name)
                cv2.imwrite(img_out, img)


def detect_from_video(detector, video_path, save_output=False, output_dir='output/'):

    cap = cv2.VideoCapture(video_path)
    if save_output:
        output_path = os.path.join(output_dir, 'detection_' + video_path.split("/")[-1])
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
            *"mp4v"), 30, (frame_width, frame_height))

    while (cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            break

        timestamp1 = time.time()
        det_boxes = detector.DetectFromImage(img)
        elapsed_time = round((time.time() - timestamp1) * 1000)  # ms
        img = detector.DisplayDetections(img, det_boxes, det_time=elapsed_time)

        cv2.imshow('TF2 Detection', img)
        if cv2.waitKey(1) == 27:
            break

        if save_output:
            out.write(img)

    cap.release()
    if save_output:
        out.release()
