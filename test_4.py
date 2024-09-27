# test_4.py
# import libs
from cv2 import (VideoCapture, imshow, waitKey, destroyAllWindows, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS)
import cv2
import numpy as np

def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    if slope == 0.0:
        slope = 1.0
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def hough_transform(image):
    rho = 1
    theta = np.pi / 180
    threshold = 20
    minLineLength = 20
    maxLineGap = 500
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)

def region_selection(image):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_colour = (255,) * channel_count
    else:
        ignore_mask_colour = 255

    rows, cols = image.shape[:2]
    bottom_right = [cols * 0.8, rows * 0.95]
    bottom_left = [cols * 0.2, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    top_right = [cols * 0.7, rows * 0.6]

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    white_polygon = cv2.fillPoly(mask, vertices, ignore_mask_colour)
    return mask

def filter_colors(image, lower_threshold):
    _, combined_mask = cv2.threshold(image, lower_threshold, 255, cv2.THRESH_BINARY)
    return combined_mask

def trapezoidal_region_selection(image, lane_lines, width_factor=0.2, height_ratio=0.4):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    rows, cols = image.shape[:2]

    bottom_center_x = cols // 2
    bottom_center_y = int(rows * 0.95)

    bottom_left = [bottom_center_x - (cols * width_factor), bottom_center_y]
    bottom_right = [bottom_center_x + (cols * width_factor), bottom_center_y]

    top_left = [0, 0]
    top_right = [0, 0]

    closest_line_x = None
    left_lane_x = []
    right_lane_x = []

    if lane_lines is not None:
        for line in lane_lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if slope < 0:
                    left_lane_x.append(x1)
                else:
                    right_lane_x.append(x1)

        min_distance = float('inf')
        for x in left_lane_x + right_lane_x:
            distance = abs(x - bottom_center_x)
            if distance < min_distance:
                min_distance = distance
                closest_line_x = x

    if closest_line_x is not None:
        bottom_left[0] = closest_line_x - (cols * width_factor)
        bottom_right[0] = closest_line_x + (cols * width_factor)

        top_left[0] = closest_line_x - (cols * width_factor)
        top_right[0] = closest_line_x + (cols * width_factor)
        top_left[1] = int(bottom_center_y - (rows * height_ratio))
        top_right[1] = int(bottom_center_y - (rows * height_ratio))
    else:
        bottom_left = [bottom_center_x - (cols * width_factor), bottom_center_y]
        bottom_right = [bottom_center_x + (cols * width_factor), bottom_center_y]
        top_left = [bottom_left[0], bottom_left[1] - (rows * height_ratio)]
        top_right = [bottom_right[0], bottom_right[1] - (rows * height_ratio)]

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)

    return mask

def process_image(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)

    polygon_mask = region_selection(blur)
    masked_image = cv2.bitwise_and(blur, blur, mask=polygon_mask)

    mean_value = cv2.mean(masked_image, mask=polygon_mask)[0]
    lower_t = max(0, mean_value + 30)

    mask = filter_colors(masked_image, lower_t)

    low_t = 50
    high_t = 150
    edges = cv2.Canny(mask, low_t, high_t)

    hough = hough_transform(edges)

    if hough is not None:  
        lane_mask = trapezoidal_region_selection(img, hough)
        result = draw_lane_lines(img, lane_lines(img, hough))
        combined_result = cv2.bitwise_and(result, result, mask=lane_mask)
    else:
        combined_result = img

    return combined_result

def main(video_file):
    input_video = VideoCapture(video_file)

    if not input_video.isOpened():
        print("Error opening video file")

    while input_video.isOpened():
        ret, frame = input_video.read()
        if ret:
            output_video = process_image(frame)
            imshow('Lane Assistance!', output_video)

        if waitKey(25) == 27:
            break

    input_video.release()

destroyAllWindows()
main('test_1.mp4')
