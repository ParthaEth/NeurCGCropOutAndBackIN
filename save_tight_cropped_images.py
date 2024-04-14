import os
import csv
import cv2
import dlib
import tqdm
import imutils
from imutils import face_utils
import numpy as np
import argparse
from utils import dlib_rect_to_bounding_box
import matplotlib.pyplot as plt
from kalman_filter_for_2d_points import ExtendedKalmanFilter

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-d", "--directory", required=True, help="path to directory containing images")
ap.add_argument("-o", "--output", required=True, help="path to output file and dir")
args = vars(ap.parse_args())

max_images = 1001
skip_n = 1  # skips every odd numbered frames
use_kf = True
fps = 30
bb_w, bb_h = 120, 120

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
        num_points = 1  # Just the top left of the bounding box
        kf = ExtendedKalmanFilter(num_points=num_points, dt=1/fps, std_acc=0.01, std_meas=0.2)

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
                (left, top, right, bott) = dlib_rect_to_bounding_box(rect, bb_w, bb_h)

                if j == 0 and use_kf:
                    init_state = np.zeros((num_points * 2 + 4, 1))  # two velocity and 2 acceleration
                    init_state[:num_points * 2, 0] = np.array([left, top])
                    kf.set_state(init_state)
                elif use_kf:
                    kf.predict()
                    kf.update(np.array([[left], [top]]))
                    left, top = kf.get_state()[:2*num_points].astype(int)
                    left, top = int(left), int(top)
                    right, bott = left + bb_w, top + bb_h

                # Store bounding box and landmarks in CSV
                landmarks = ';'.join(f"{x},{y}" for (x, y) in shape)
                csv_writer.writerow([filename, i + 1, f"{left},{top},{right},{bott}", landmarks])
                cv2.imwrite(os.path.join(args["output"][:-4], filename), image[top:bott, left:right])
            #     for (x, y) in shape:
            #         cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            # plt.imshow(image[int(top):int(bott), int(left):int(right), ::-1])
            # plt.show()
        if j >= max_images:
            break
