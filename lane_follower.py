import cv2
import numpy as np
import math


class LaneFollower(object):
    def __init__(self, robot=None):
        self.robot = robot
        self.curr_steering_angle = 90

    def follow_lane(self, frame):

        lane_lines, frame = detect_lane(frame)
        final_frame = self.steer(frame, lane_lines)

        return final_frame

    def steer(self, frame, lane_lines):

        if len(lane_lines) == 0:
            print('No lines detected, nothing to do...')
            return frame
        new_steering_angle = get_steering_angle(frame, lane_lines)

        if self.robot is not None:
            print("Turn about steering angle")

        curr_heading_image = display_heading_line(frame, self.curr_steering_angle)
        return curr_heading_image


def detect_lane(frame):

    print("Detecting lane lines...")

    edges = detect_edges(frame)
    cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(cropped_edges)
    line_segments_image = display_lines(frame, line_segments)
    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)

    return lane_lines, lane_lines_image


def detect_edges(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([60, 40, 40], dtype='uint8') # lower limit of blue color
    upper_blue = np.array([150, 255, 255], dtype='uint8') # upper limit
    mask = cv2.inRange(hsv, lower_blue, upper_blue) # filter out everything but blue

    edges = cv2.Canny(mask, 200, 400)
    cv2.imshow('edges', edges)
    return edges


def region_of_interest(edges):
    height, width = edges.shape # extract the height and width of the edges frame

    mask = np.zeros_like(edges) # make an empty matrix  with the same dimensions as the edges frame

    # focus lower half of the screen
    # specify the coordinates of 4 points (lower and upper left and right)
    polygon = np.array([[
        (0, height),
        (0, height/2),
        (width, height/2),
        (width, height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255) # fill polygon with blue color
    cropped_edges = cv2.bitwise_and(edges, mask)
    cv2.imshow('roi', cropped_edges)
    return cropped_edges


def detect_line_segments(cropped_edges):
    rho = 1
    theta = np.pi / 180
    min_threshold = 10
    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold, np.array([]), minLineLength=4, maxLineGap=0)
    return line_segments


def average_slope_intercept(frame, line_segments):
    # This function combines line segments into one or two lane lines
    # If all line slopes are < 0: then we only have detected left lane
    # If all line slopes are > 0: then we only have detected right lane

    lane_lines = []

    if line_segments is None:
        print("No line segment detected")
        return lane_lines
    height, width, _ = frame.shape
    left_fit = []
    right_fit = []
    boundary = 1/3

    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                print("skipping vertical lines (slope = infinity)")
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

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

    # lane_lines is a 2-D array consisting the coordinates of the right and left lane lines
    # for example: lane_lines = [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    # where the left array is for left lane and the right array is for right lane
    # all coordinate points are in pixels
    return lane_lines


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height # bottom of the frame
    y2 = int(y1 / 2) # make points from middle of the frame down

    if slope == 0:
        slope = 0.1

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


def get_steering_angle(frame, lane_lines):
    height, width, _ = frame.shape

    if len(lane_lines) == 2: # if 2 lines are detected
        _, _, left_x2, _ = lane_lines[0][0] # extract left x2 from lane_lines array
        _, _, right_x2, _ = lane_lines[1][0] # extract right x2 from lane_lines array
        camera_mid_offset_percent = 0.02 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(width / 2 * ( + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)

    elif len(lane_lines) == 1: # only one line detected
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)

    elif len(lane_lines) == 0: # no line detected
        print("No lane lines detected")
        return -90

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    return steering_angle


def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=5, max_angle_deviation_one_lane=1):
    # Using last steering angle to stabilize the steering angle
    # this can be improved to use last N angles, etc
    # if new angle is too diffrent from current angle, only turn by max_angle_deviation_angle

    if num_of_lane_lines == 2:
        # if both lane lines detected, then we can derive more
        max_angle_deviation = max_angle_deviation_two_lines
    else:
        max_angle_deviation = max_angle_deviation_one_lane

    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle

    return stabilized_steering_angle


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=10):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def display_heading_line(frame, steering_angle, line_color=(0,0,255), line_width=5):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right

    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)

    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image



