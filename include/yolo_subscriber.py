#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Subscribing bounding boxes of the detected objects published by the `darknet_ros` repo.
'''

from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
import rospy
import cv2
import numpy as np

''' BoundingBoxes:
Header header
Header image_header
BoundingBox[] bounding_boxes
'''

''' BoundingBox:
float64 probability
int64 xmin
int64 ymin
int64 xmax
int64 ymax
int16 id
string Class
'''


def in_which_box(xy, bboxes):
    x, y = xy[0], xy[1]
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
            return i
    return -1


def draw_box(img_disp, bbox, color=[255, 0, 0], thickness=2):
    xmin, ymin, xmax, ymax = bbox
    xmin, ymin, width, height = map(
        int, [xmin, ymin, xmax-xmin, ymax-ymin])  # to int
    bbox = (xmin, ymin, width, height)
    img_disp = cv2.rectangle(
        img_disp, rec=bbox, color=color, thickness=thickness)
    return img_disp


class YoloSubscriber(object):
    def __init__(
            self,
            topic_name="darknet_ros/bounding_boxes",
            target_classes=["cup", "bowl", "banana", "laptop"],
            # Only the bboxes of the target classes will be accepted.
    ):
        self._detect_results = None
        self._sub = rospy.Subscriber(
            topic_name, BoundingBoxes, self._callback)
        self._target_classes = set(target_classes)

    def get_bboxes(self):

        # -- Get data.
        while (self._detect_results is None) and not rospy.is_shutdown():
            rospy.sleep(0.001)
        detect_results = self._detect_results
        self._detect_results = None  # Reset data.

        # -- Process the bounding boxes.
        res = []
        bboxes = detect_results.bounding_boxes
        for bbox in bboxes:
            if bbox.Class in self._target_classes:
                res.append(
                    [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax])
        return res

    def _callback(self, _detect_results):
        self._detect_results = _detect_results


def test_subscriber_and_draw_bbox():
    subscriber = YoloSubscriber(
        topic_name="darknet_ros/bounding_boxes",
        target_classes=["cup", "bowl", "banana", "laptop"])
    xy = (320, 450)  # A point. We want to check if it's in a bounding box.
    img_disp0 = 255 + np.zeros((480, 640, 3), np.uint8)

    while not rospy.is_shutdown():
        bboxes = subscriber.get_bboxes()
        ith_box = in_which_box(xy, bboxes)
        print("Bobes:" + str(bboxes))
        print("Point (x={}, y={}): ".format(xy[0], xy[1]))
        img_disp = img_disp0.copy()
        if ith_box >= 0:
            print("  In {}th bbox".format(ith_box))
            img_disp = draw_box(img_disp, bboxes[ith_box])
            cv2.circle(img_disp, xy, radius=3, color=[0, 0, 255], thickness=-1)
        else:
            print("Not in any boxes.")
            assert(False)
        cv2.imshow("Which box the object is in.", img_disp)
        cv2.waitKey(10)


if __name__ == '__main__':
    node_name = "yolo_subscriber"
    rospy.init_node(node_name)
    rospy.sleep(0.1)
    test_subscriber_and_draw_bbox()
    rospy.logwarn("Node `{}` stops.".format(node_name))
