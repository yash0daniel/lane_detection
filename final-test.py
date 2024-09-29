# Import necessary libraries
from cv2 import (VideoCapture, imshow, waitKey, destroyAllWindows)
import cv2
import numpy as np

def average_slope_intercept(lines):
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:  # Ignore vertical lines
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            # Filter for appropriate slopes to identify lane lines
            if -0.8 > slope > -2.0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            elif 0.8 < slope < 2.0:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return ((x1, int(y1)), (x2, int(y2)))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]  # Bottom of the image
    y2 = int(y1 * 0.78)  # Slightly lower than the middle
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=10):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def hough_transform(image):
    rho = 2
    theta = np.pi / 180
    threshold = 50
    minLineLength = 50
    maxLineGap = 160
    return cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

def region_selection(image, curve_factor=0.5):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_colour = (255,) * channel_count
    else:
        ignore_mask_colour = 255

    rows, cols = image.shape[:2]
    
    # Calculate dynamic points for flexibility
    bottom_right = [cols * 0.8, rows * 0.95]
    bottom_left = [cols * 0.2, rows * 0.95]
    
    # Adjust the top points based on the curve_factor
    top_left_x = cols * (0.4 - curve_factor * 0.1)  # adjustments
    top_right_x = cols * (0.6 + curve_factor * 0.1)  # adjustments
    top_left = [top_left_x, rows * 0.7]
    top_right = [top_right_x, rows * 0.7]

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_colour)

    return mask

def filter_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # imshow('hsv filter_colors', hsv)

    # Filter white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Filter yellow color in HSV
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine all masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    return cv2.bitwise_and(image, image, mask=combined_mask)

def remove_shadows(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # imshow('hsv', hsv)
    h, s, v = cv2.split(hsv)
    _, v_thresh = cv2.threshold(v, 60, 255, cv2.THRESH_BINARY)
    v_filtered = cv2.merge((h, s, v_thresh))
    return cv2.cvtColor(v_filtered, cv2.COLOR_HSV2BGR)

def full_scan(frame):
    # A more extensive edge detection method if lanes are not detected properly
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    imshow('edges', edges)
    lines = hough_transform(edges)
    
    # Check if lines are detected, if not, use full scan
    if lines is None:
        # print("No lines detected, performing full scan.")
        return frame
    
    lanes = lane_lines(frame, lines)
    frame_with_lanes = draw_lane_lines(frame, lanes)
    return frame_with_lanes

def process_image(frame):
    # Remove shadows
    shadow_removed = remove_shadows(frame)
    # imshow('shadow_removed', shadow_removed)

    # Filter for road colors (white and yellow lines)
    filtered = filter_colors(shadow_removed)
    # imshow('filtered', filtered)

    # Apply region selection to limit area of interest
    region_mask = region_selection(filtered)
    # imshow('region_mask', region_mask)
    masked_image = cv2.bitwise_and(filtered, region_mask)
    # imshow('masked_image', masked_image)

    # Convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    # imshow('gray', gray)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # imshow('blur', blur)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)
    imshow('edges', edges)

    # Hough transform for lane lines
    lines = hough_transform(edges)

    # Draw lane lines
    if lines is not None and len(lines) > 2:
        lanes = lane_lines(frame, lines)
        frame_with_lanes = draw_lane_lines(frame, lanes)
    else:
        # print("Lanes are not detected properly. Switching to full scan.")
        frame_with_lanes = full_scan(frame)

    # Show the masked image and lane detection result
    imshow('Lane Detection', frame_with_lanes)
    return frame_with_lanes

def main(video_file):
    cap = VideoCapture(video_file)

    if not cap.isOpened():
        print("Error opening video file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_image(frame)

        if waitKey(25) == 27:  # Esc key to exit
            break

    cap.release()
    destroyAllWindows()

main('test_1.mp4')
