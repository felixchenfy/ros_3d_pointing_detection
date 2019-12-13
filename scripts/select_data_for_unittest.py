'''
From the whole dataset, select the images for unit test.
Besides, merge two test images into one image -- 
    one on the left side, the other on the right side, 
    so that there are two people in the output image,
    in order to test whether the program supports multiple people.
'''


import cv2
import numpy as np
import glob
import os
import sys

if True:  # Add project root
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+'/../'
    sys.path.append(ROOT)

if False:
    SRC_FOLDER = ROOT + "data/12-12-11-58-56-290/"
    START_IDX = 205
    END_IDX = 430
elif False:
    SRC_FOLDER = ROOT + "data/12-12-19-31-38-880/"
    START_IDX = 405
    END_IDX = 600
else:
    SRC_FOLDER = ROOT + "data/12-12-19-52-07-396/"
    START_IDX = 730
    END_IDX = 890

SKIP = 3
DST_FOLDER_COLOR = ROOT + "output/color/"
DST_FOLDER_DEPTH = ROOT + "output/depth/"


def read_ith_image(i):
    color = cv2.imread(
        SRC_FOLDER + "color/{:05d}.png".format(i))
    depth = cv2.imread(
        SRC_FOLDER + "depth/{:05d}.png".format(i), cv2.IMREAD_UNCHANGED)
    return color, depth


def makedir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':

    makedir(DST_FOLDER_DEPTH)
    makedir(DST_FOLDER_COLOR)
    L = END_IDX - START_IDX + 1
    i = 0
    for cnt in range(0, L, SKIP):
        print("Processing the {}/{}th image".format(cnt+1, L))
        image_name = START_IDX + cnt
        color, depth = read_ith_image(image_name)
        filename = "{:05d}.png".format(i)
        fcolor = DST_FOLDER_COLOR+filename
        fdepth = DST_FOLDER_DEPTH+filename
        cv2.imwrite(fcolor, color)
        cv2.imwrite(fdepth, depth)
        print("  Write color image to: " + fcolor)
        print("  Write depth image to: " + fdepth)
        i += 1
