#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
                    [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
                )
        return res

    def _callback(self, _detect_results):
        self._detect_results = _detect_results


def test_subscribe_and_draw_bbox():
    sub = YoloSubscriber(
        topic_name="darknet_ros/bounding_boxes",
        target_classes=["cup", "bowl", "banana", "laptop"])
    list_xy = [
        [250, 400],
        [280, 400],
        [320, 400],
        [350, 400],
        [380, 400],
        [450, 400],
    ]
    img_disp0 = 255 + np.zeros((480, 640, 3), np.uint8)

    while not rospy.is_shutdown():
        bboxes = sub.get_bboxes()
        print(bboxes)
        ith_box = -1
        for xy in list_xy:
            ret = in_which_box(xy, bboxes)
            if ret >= 0:
                ith_box = ret
                print("In {}th bbox".format(ith_box))
                break
        print("Complete an image.")
        img_disp = img_disp0.copy()
        img_disp = draw_box(img_disp, bboxes[ith_box])
        cv2.imshow("bbox", img_disp)
        cv2.waitKey(10)


if __name__ == '__main__':
    node_name = "yolo_subscriber"
    rospy.init_node(node_name)
    rospy.sleep(0.1)
    test_subscribe_and_draw_bbox()
    rospy.logwarn("Node `{}` stops.".format(node_name))
