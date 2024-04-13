import numpy as np
import cv2


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def draw_rotated_box(image, dest_box, color=(0, 255, 0), thickness=2):
    """
    Draws a rotated box on the image with the given coordinates.

    Parameters:
    image (numpy.ndarray): The image on which to draw the box.
    dest_box (list of tuples): List of coordinates as [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
                               The coordinates should be ordered as [bottom-left, bottom-right,
                               top-right, top-left].
    color (tuple): The color of the box in BGR format (default is green).
    thickness (int): The thickness of the lines of the box (default is 2).

    Returns:
    numpy.ndarray: The image with the drawn box.
    """
    # Ensure the points are in integer format
    pts = np.array(dest_box, np.int32)

    # Reshape points in a form required by polylines
    pts = pts.reshape((-1, 1, 2))

    # Draw the polygon
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

    return image


def draw_points(image, points, color=(0, 0, 255), radius=3, thickness=-1):
    """
    Draws points on an image.

    Parameters:
    image (numpy.ndarray): The image on which to draw the points.
    points (list of tuples): List of coordinates as [(x1, y1), (x2, y2), ...] for each point.
    color (tuple): The color of the points in BGR format (default is red).
    radius (int): Radius of each point circle (default is 3 pixels).
    thickness (int): Thickness of the circle outline. If -1, it will fill the circle (default is filled).

    Returns:
    numpy.ndarray: The image with the drawn points.
    """
    # Iterate over all points
    for point in points:
        # Ensure the point is in integer format
        center = tuple(np.int32(point))
        # Draw each point as a circle
        cv2.circle(image, center, radius, color, thickness)

    return image


def dlib_rect_to_bounding_box(rect, target_width, target_height):
    """
    Adjusts the bounding box to a specific width and height while keeping the center the same.

    Parameters:
    rect (dlib.rectangle): The original bounding box.
    target_width (int): The desired width of the bounding box.
    target_height (int): The desired height of the bounding box.

    Returns:
    left, top, right, bottom: A new bounding box of the specified size centered around the original rect's center.
    """
    center_x = rect.left() + rect.width() // 2
    center_y = rect.top() + rect.height() // 2

    new_left = max(0, center_x - target_width // 2)
    new_top = max(0, center_y - target_height // 2)
    new_right = new_left + target_width
    new_bottom = new_top + target_height

    return new_left, new_top, new_right, new_bottom