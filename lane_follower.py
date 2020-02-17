import cv2
import numpy as np
import math
import sys
import time


def detect_edges(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([30, 40, 0], dtype="uint8")
    upper_blue = np.array([150, 255, 255], dtype="uint8")
    # blue color mask
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # detect edges
    edges = cv2.Canny(mask_blue, 200, 400)

    return edges


def detect_red_edges(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # filter for red lines
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    # red mask
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    edges_red = cv2.Canny(mask_red, 50, 100)
    return edges_red


def region_of_interest(edges):
    height, width = edges.shape  # extract the height and width of the edges frame
    mask = np.zeros_like(edges)  # make an empty matrix with same dimensions of the edges frame

    # only focus lower half of the screen
    # specify the coordinates of 4 points (lower left, upper left, upper right, lower right)
    polygon = np.array([[
        (0, height),
        (0, height / 2),
        (width, height / 2),
        (width, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)  # fill the polygon with blue color

    cropped_edges = cv2.bitwise_and(edges, mask)
    # cv2.imshow("roi",cropped_edges)  take the edged frame as parameter and draws a polygon with 4 preset points

    return cropped_edges


def detect_line_segments(cropped_edges):
    rho = 1  # the the distance precision in pixels
    theta = np.pi / 180  # angular precision in radians
    min_threshold = 10  # minimum vote it should get for it to be considered as a line

    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold,
                                    np.array([]), minLineLength=5, maxLineGap=150)

    return line_segments


# takes the frame under processing and lane segments detected using
# Hough transform and returns the average slope and intercept of two lane lines
def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None or len(line_segments) == 2:
        print("Go up")
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                print("skipping vertical lines (slope = infinity")
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)

            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines


# helper function for average_slope_intercept() function which will return
# the bounded coordinates of the lane lines
# (from the bottom to # the middle of the frame)
def make_points(frame, line):
    height, width, _ = frame.shape

    slope, intercept = line

    y1 = height  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down

    if slope == 0:  # slope = 0 which means y1 = y2 (horizontal line)
        slope = 0.1

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]


# display the lane lines on the frames
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):
    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)

    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    return line_image


# If steering_angle = 90, it means that the car has a heading line perpendicular
# to "height / 2" line and the car will move forward without steering.
# If steering_angle > 90, the car should steer to right otherwise
# it should steer left
def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle / 180.0 * math.pi

    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    # combine two images but with giving each one a weight
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


def get_steering_angle(frame, lane_lines):
    height, width, _ = frame.shape

    if len(lane_lines) == 2:  # if two lane lines are detected
        _, _, left_x2, _ = lane_lines[0][0]  # extract left x2 from lane_lines array
        _, _, right_x2, _ = lane_lines[1][0]  # extract right x2 from lane_lines array
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)

    elif len(lane_lines) == 1:  # if only one line is detected
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)

    elif len(lane_lines) == 0:  # if only no line is detected
        x_offset = 0
        y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    return steering_angle


video = cv2.VideoCapture(1)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

time.sleep(1)
lastTime = 0

kp = 0.4
kd = kp * 0.65

while True:
    k = cv2.waitKey(1) & 0xFF
    ret, frame = video.read()
    frame = cv2.flip(frame, -1)

    cv2.imshow("original", frame)
    edges = detect_edges(frame)
    roi = region_of_interest(edges)
    line_segments = detect_line_segments(roi)
    edges_2 = detect_red_edges(frame)
    roi_2 = region_of_interest(edges_2)
    line_segments_2 = detect_line_segments(roi_2)
    if line_segments_2 is not None:
        print("Go down!")
    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    steering_angle = get_steering_angle(frame, lane_lines)
    heading_image = display_heading_line(lane_lines_image, steering_angle)
    cv2.imshow("heading line", heading_image)

    now = time.time()
    dt = now - lastTime

    deviation = steering_angle - 90
    error = abs(deviation)

    if deviation < 5 and deviation > -5:
        deviation = 0
        error = 0
        print("do not steer if there is a 10-degree error range - move forward")

    elif deviation > 5:
        print("steer right if the deviation is positive")

    elif deviation < -5:
        print("steer left if the deviation is positive")

    lastError = error
    lastTime = time.time()
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()





