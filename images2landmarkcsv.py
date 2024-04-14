import os
import csv
import cv2
import dlib
import tqdm
import imutils
from imutils import face_utils
import numpy as np
import argparse
import matplotlib.pyplot as plt
from kalman_filter_for_2d_points import ExtendedKalmanFilter

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-d", "--directory", required=True, help="path to directory containing images")
ap.add_argument("-o", "--output", required=True, help="path to output CSV file")
args = vars(ap.parse_args())

max_images = 1001
skip_n = 0  # skips every odd numbered frames
use_kf = True  # use kalman filtering
fps = 30

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Prepare CSV file to store results
with open(args["output"], mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(["Image", "Face_ID", "Bounding_Box", "Landmarks"])

    # Iterate over each image in the directory
    files_to_process = os.listdir(args["directory"])
    files_to_process = sorted(files_to_process)[0:len(files_to_process):skip_n+1]

    if use_kf:
        num_points = 68  # 68 keypoints
        kf = ExtendedKalmanFilter(num_points=num_points, dt=1/fps, std_acc=1, std_meas=0.1)

    for j, filename in enumerate(tqdm.tqdm(files_to_process)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(args["directory"], filename)
            image = cv2.imread(image_path)
            # image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            for (i, rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                (x, y, w, h) = face_utils.rect_to_bb(rect)

                if j == 0 and use_kf:
                    init_state = np.zeros((num_points * 2 + 4, 1))
                    init_state[:num_points*2, 0] = shape.flatten()

                    kf.set_state(init_state)
                elif use_kf:
                    kf.predict()
                    kf.update(shape.flatten()[..., None])
                    shape = kf.get_state()[: 2*num_points].reshape(num_points, 2).astype(int)
                # Store bounding box and landmarks in CSV
                landmarks = ';'.join(f"{x},{y}" for (x, y) in shape)
                csv_writer.writerow([filename, i + 1, f"{x},{y},{w},{h}", landmarks])
            #     for (x, y) in shape:
            #         cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            # plt.imshow(image[:, :, ::-1])
            # plt.show()
        if j >= max_images:
            break
