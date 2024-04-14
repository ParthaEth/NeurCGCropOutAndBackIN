import os
import cv2
import tqdm
import numpy as np
import pandas as pd
from compute_transform import transform_s2dest, apply_transformation
from utils import draw_rotated_box, draw_points
import matplotlib.pyplot as plt


def parse_landmarks(landmark_string):
    # Convert the semicolon-separated string of coordinates into a numpy array
    return np.array([list(map(int, point.split(','))) for point in landmark_string.split(';')])


def apply_transformation_to_image(image, R, scale, translation):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), np.degrees(np.arccos(R[0, 0])), scale)
    M[:, 2] += translation  # Adding translation to the transformation matrix
    transformed_image = cv2.warpAffine(image, M, (cols, rows))
    return transformed_image


def overlay_images(image_1, image_2, dest_box, exclude_padding=2):
    """
    Alpha blends image_2 into a rotated bounding box of image_1 with a gradient alpha from center to edges.

    Parameters:
    image_1 (numpy.ndarray): Destination image where image_2 will be blended.
    image_2 (numpy.ndarray): Source image to blend into image_1.
    dest_box (list): Coordinates of the destination rotated box as [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
                      The coordinates should be ordered as [bottom-left, bottom-right, top-right, top-left].

    Returns:
    numpy.ndarray: The resulting image with image_2 blended into image_1 at the specified rotated box.
    """
    # Ensure the destination box is a numpy array of float points
    dest_box = np.array(dest_box, dtype=np.float32)

    # Define the source points from the dimensions of image_2
    h, w = image_2.shape[:2]
    src_points = np.array([[exclude_padding, h-exclude_padding],
                           [w - exclude_padding, h - exclude_padding],
                           [w - exclude_padding, exclude_padding],
                           [exclude_padding, exclude_padding]], dtype=np.float32)

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dest_box)

    # Warp image_2 to the coordinates of the destination box
    transformed_image_2 = cv2.warpPerspective(image_2, matrix, (image_1.shape[1], image_1.shape[0]))

    # Create a mask for the transformed area
    mask = np.zeros_like(transformed_image_2, dtype=np.uint8)
    cv2.fillConvexPoly(mask, dest_box.astype(int), (255,)*image_2.shape[2], lineType=cv2.LINE_AA)

    # Generate a radial gradient mask
    center_x, center_y = np.mean(dest_box[:, 0]), np.mean(dest_box[:, 1])
    Y, X = np.ogrid[:image_1.shape[0], :image_1.shape[1]]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.max(dist_from_center)/7
    gradient_mask = np.clip(1 - np.power(dist_from_center / max_dist, 3), 0, 1)

    # Use the gradient mask to blend the transformed image and the mask
    alpha_mask = (mask / 255.0) * gradient_mask[..., np.newaxis]

    # Blending using the alpha mask
    for c in range(0, 3):
        image_1[..., c] = alpha_mask[..., 0] * transformed_image_2[..., c] + \
                          (1 - alpha_mask[..., 0]) * image_1[..., c]

    return image_1


def main():
    # In this case we assume the rectangular region to paste back is axis aligned into the destination frame.
    # We won't compute the rotation or scale!
    aligned_paste = True
    # aligned_paste = False
    # source = 'acharjo_neurCG'
    source = 'ipman_kf'
    destination = 'life-scn-ch4-v2'
    if aligned_paste:
        source_landmarks = pd.read_csv(f'{source}_src_kpt.csv')
    else:
        source_landmarks = pd.read_csv(f'{source}.csv')

    source_landmarks.sort_values(by='Image', ascending=True, inplace=True)
    dest_landmarks = pd.read_csv(f'{destination}.csv')
    dest_landmarks.sort_values(by='Image', ascending=True, inplace=True)

    for i, (index, row_src_csv) in enumerate(tqdm.tqdm(source_landmarks.iterrows())):
        row2 = dest_landmarks.iloc[i]

        if aligned_paste:
            R = np.eye(2)
            scale = 1.0
            translation = parse_landmarks(row_src_csv['Bounding_Box'])[0][:2]
        else:
            # Compute tehrelative transform necessary
            source_lm = parse_landmarks(row_src_csv['Landmarks'])
            dest_lm = parse_landmarks(row2['Landmarks'])

            transform = transform_s2dest(source_lm, dest_lm)
            R, scale, translation = transform['rotation'], transform['scale'], transform['translation']

        # Read images
        # src_image_path = os.path.join(source, row_src_csv['Image'])
        src_image_path = os.path.join(source, f'frame_{(i+1):04d}.png')
        dest_image_path = os.path.join(destination, row2['Image'])
        src_image = cv2.imread(src_image_path)
        dest_image = cv2.imread(dest_image_path)

        h, w = src_image.shape[:2]
        exclude_padding = 2  #px
        dest_box = np.array([[exclude_padding, h - exclude_padding],
                             [w - exclude_padding, h - exclude_padding],
                             [w - exclude_padding, exclude_padding],
                             [exclude_padding, exclude_padding]], dtype=np.float32)
        dest_box = apply_transformation(dest_box, R, scale, translation)
        final_image = overlay_images(dest_image, src_image, dest_box, exclude_padding)
        # final_image = draw_rotated_box(dest_image, dest_box)
        # final_image = draw_points(final_image, dest_lm)
        output_path = os.path.join('NeurCG_fullbody_kf', row_src_csv['Image'])
        cv2.imwrite(output_path, final_image)
        # plt.imshow(final_image[:, :, ::-1])
        # plt.show()
        # print(f"Processed and saved: {output_path}")


if __name__ == "__main__":
    main()
