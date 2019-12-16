#!/usr/bin/env python
# -*- coding: utf-8 -*-

from calc_3d_dist import point_3d_line_distance, point_plane_distance
from yolo_subscriber import YoloSubscriber, in_which_box, draw_box
import numpy as np
import cv2
import rospy
import argparse
import glob
import time
import math

if True:  # Add project root
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+'/'


if True:  # Import scripts from another ROS repo.
    sys.path.append(ROOT + "../ros_openpose_rgbd")
    from lib_draw_3d_joints import Human, set_default_params
    from lib_openpose_detector import OpenposeDetector
    from utils.lib_rgbd import RgbdImage, MyCameraInfo
    from utils.lib_ros_rgbd_pub_and_sub import ColorImageSubscriber, DepthImageSubscriber, CameraInfoSubscriber
    from utils.lib_ros_rgbd_pub_and_sub import ColorImagePublisher
    from utils.lib_geo_trans import rotx, roty, rotz, get_Rp_from_T, form_T
    from utils.lib_rviz_marker import RvizMarker

''' ------------------------------ Settings ------------------------------------ '''
TOPIC_RES_IMAGE = "3d_pointing/res_image"
DST_RES_IMAGE_VIZ = "output/res_img/"
YOLO_TOPIC_NAME = "darknet_ros/bounding_boxes"
OBJECT_CLASSES = ["cup", "bowl", "banana", "laptop"]

VIZ_HUMAN_ID = 1
VIZ_ID_RAY = 10000001
VIZ_ID_HIT_POINT = 10000002
LINE_WIDTH_2D_RAY = 3
DOT_RADIUS_2D_HIT_POINT = 8

''' ------------------------------ Command line inputs ------------------------------------ '''


def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description="Detect human joints and then draw in rviz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # -- Select data source.
    parser.add_argument(
        "-s", "--data_source",
        default="rostopic",
        choices=["rostopic", "disk"],
        help="The option `disk` is only for debug. "
        "Since darknet_ros(YOLO) package reads from rostopic, "
        "we need to set this as `rostopic` when not debugging.")
    parser.add_argument(
        "-y", "--detect_object", type=Bool,
        default=True,
        help="If this is True, you must have already started the object detection by "
        "`roslaunch ros_3d_pointing_detection darknet_ros.launch`. "
        "Set this to False when you are debugging.")
    parser.add_argument(
        "-z", "--detect_hand", type=Bool,
        default=False,
        help="The hand joints are not used here. Besides, it's very slow.")
    parser.add_argument(
        "-u", "--depth_unit", type=float,
        default="0.001",
        help="Depth is (pixel_value * depth_unit) meters "
        "at each pixel of the depth image.")
    parser.add_argument(
        "-r", "--is_using_realsense", type=Bool,
        default=False,
        help="If the data source is Realsense, set this to true. "
        "Then, the drawn joints will change the coordinate to be the same as "
        "Realsense's point cloud. The reason is,"
        "I used a different coordinate direction than Realsense."
        "(1) For me, I use X-Right, Y-Down, Z-Forward,"
        "which is the convention for camera."
        "(2) For Realsense ROS package, it's X-Forward, Y-Left, Z-Up.")

    # -- "rostopic" as data source.
    parser.add_argument("-a", "--ros_topic_color",
                        default="camera/color/image_raw")
    parser.add_argument("-b", "--ros_topic_depth",
                        default="camera/aligned_depth_to_color/image_raw")
    parser.add_argument("-c", "--ros_topic_camera_info",
                        default="camera/color/camera_info")

    # -- "disk" as data source.
    parser.add_argument("-d", "--base_folder",
                        default=ROOT,
                        help="If reading from disk and "
                        "if the input path is a relative path, "
                        "base_folder will be preappend to the relative path.")
    parser.add_argument("-e", "--folder_color",
                        default="data/images_n76/color/")
    parser.add_argument("-f", "--folder_depth",
                        default="data/images_n76/depth/")
    parser.add_argument("-g", "--camera_info_file",
                        default="data/images_n76/cam_params_realsense.json")

    # -- Get args.
    inputs = rospy.myargv()[1:]
    inputs = [s for s in inputs if s.replace(" ", "") != ""]  # Remove blanks.
    args = parser.parse_args(inputs)

    # -- Deal with relative path.
    b = args.base_folder + "/"
    args.folder_color = b + args.folder_color
    args.folder_depth = b + args.folder_depth
    args.camera_info_file = b + args.camera_info_file

    # -- Return
    return args


def Bool(v):
    ''' A bool class for argparser '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


''' ------------------------------ Data loader ------------------------------------ '''


class DataReader_DISK(object):
    def __init__(self, args):
        self._fcolors = sorted(glob.glob(args.folder_color + "/*"))
        basenames = [os.path.basename(s) for s in self._fcolors]
        self._fdepths = [args.folder_depth + "/" + s for s in basenames]
        self._camera_info = MyCameraInfo(
            camera_info_file_path=args.camera_info_file)
        self._depth_unit = args.depth_unit
        self._cnt_imgs = 0
        self._total_images = len(self._fcolors)

    def total_images(self):
        return self._total_images

    def read_next_data(self):
        def read_img(folders, i):
            return cv2.imread(folders[i], cv2.IMREAD_UNCHANGED)
        color = read_img(self._fcolors, self._cnt_imgs)
        depth = read_img(self._fdepths, self._cnt_imgs)
        self._cnt_imgs += 1
        rgbd = RgbdImage(color, depth,
                         self._camera_info,
                         depth_unit=self._depth_unit)
        return rgbd


class DataReader_ROS(object):
    def __init__(self, args):
        self._sub_c = ColorImageSubscriber(args.ros_topic_color)
        self._sub_d = DepthImageSubscriber(args.ros_topic_depth)
        self._sub_i = CameraInfoSubscriber(args.ros_topic_camera_info)
        self._depth_unit = args.depth_unit
        self._camera_info = None
        self._cnt_imgs = 0

    def total_images(self):
        ''' Set a large number here. '''
        return 9999

    def read_next_data(self):
        depth = self._read_depth()
        color = self._read_color()
        camera_info = self._get_camera_info()
        self._cnt_imgs += 1
        rgbd = RgbdImage(color, depth,
                         camera_info,
                         depth_unit=self._depth_unit)
        return rgbd

    def _get_camera_info(self):
        '''
        Since a camera parameter won't change (usually),
        we read it from cache after it's initialized.
        '''
        if self._camera_info is None:
            while (not self._sub_i.has_camera_info()) and (not rospy.is_shutdown):
                rospy.sleep(0.001)
            if self._sub_i.has_camera_info:
                self._camera_info = MyCameraInfo(
                    ros_camera_info=self._sub_i.get_camera_info())
        return self._camera_info

    def _read_depth(self):
        while not self._sub_d.has_image() and (not rospy.is_shutdown()):
            rospy.sleep(0.001)
        depth = self._sub_d.get_image()
        return depth

    def _read_color(self):
        while not self._sub_c.has_image() and (not rospy.is_shutdown()):
            rospy.sleep(0.001)
        color = self._sub_c.get_image()
        return color


''' ------------------------------ Math ------------------------------------ '''


def cam_to_world(xyz_in_camera, camera_pose):
    xyz_in_world = camera_pose.dot(np.append(xyz_in_camera, [1.0]))[0:3]
    return xyz_in_world


def cam2pixel(xyz_in_camera, camera_intrinsics):
    ''' Project a point represented in camera coordinate onto the image plane.
    Arguments:
        xyz_in_camera {np.ndarray}: (3, ).
        camera_intrinsics {np.ndarray}: 3x3.
    Return:
        xy {np.ndarray, np.float32}: (2, ). Column and row index.
    '''
    pt_3d_on_cam_plane = xyz_in_camera/xyz_in_camera[2]  # z=1
    xy = camera_intrinsics.dot(pt_3d_on_cam_plane)[0:2]
    xy = tuple(int(v) for v in xy)
    return xy


''' ------------------------------ 3D pointing detection ------------------------------------ '''


def get_joints_for_pointing(arm_3_joints_xyz):
    ''' Use two joints as the pointing direction of the arm.
    Here I select the shoulder and wrist, which are 0 and 2. 
    Some comments in other functions are written based on this setting.
    '''
    p1, p2 = arm_3_joints_xyz[0].copy(), arm_3_joints_xyz[2].copy()
    return p1, p2


def is_arm_stretched(
    arm_3_joints_xyz,
    angle_thresh=30.0,  # degrees
):
    ''' If the angle between upper arm and forearm is smaller than angle_thresh,
    then the arm is stretched. 
    Arguments:
        arm_3_joints_xyz {np.ndarray}: shape=(3, 3). Three joints' xyz positions.
    Return:
        is_stretched {bool}.
    '''
    p0, p1, p2 = arm_3_joints_xyz[0], arm_3_joints_xyz[1], arm_3_joints_xyz[2]
    vec1 = np.array(p1 - p0)
    vec2 = np.array(p2 - p1)
    angle = np.arccos(
        vec1.dot(vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    angle = angle / math.pi * 180.0
    is_stretched = np.abs(angle) <= angle_thresh
    return is_stretched


def get_3d_ray_hit_point(
    arm_3_joints_xyz, pcl_xyz,
    thresh_in_front_of_wrist=0.50,  # >= this
    thresh_close_to_line=0.1,  # <= this
):
    ''' Get the hit point between the pointing ray and the point cloud.

    A point in the point cloud that is
        (1) in front of the wrist for `thresh_in_front_of_wrist`,
        (2) and is close to the ray within `thresh_close_to_line`
    is considered as the hit point.

    Arguments:
        arm_3_joints_xyz {np.ndarray}: shape=(3, 3). Three joints' xyz positions.
        pcl_xyz {np.ndarray}: shape=(N, 3). N points of xyz positions.
    Return:
        ret {bool}: Is there a valid hit point.
        xyz {np.ndarray}: shape=(3, ). The hit point's position.
    '''
    p1, p2 = get_joints_for_pointing(arm_3_joints_xyz)
    valid_pts = pcl_xyz

    # Select points that are in front of the wrist.
    dists_plane = point_plane_distance(valid_pts, p1, p2-p1)
    thresh = thresh_in_front_of_wrist + np.linalg.norm(p2-p1)
    valid_idx = dists_plane >= thresh
    valid_pts = valid_pts[valid_idx]
    dists_plane = dists_plane[valid_idx]
    if valid_pts.size == 0:
        return False, None

    # Select points that are close to the pointing direction.
    dists_3d_line = point_3d_line_distance(valid_pts, p1, p2)
    valid_idx = dists_3d_line <= thresh_close_to_line
    valid_pts = valid_pts[valid_idx]
    if valid_pts.size == 0:
        return False, None
    dists_plane = dists_plane[valid_idx]

    # Get hit point.
    closest_point_idx = np.argmin(dists_plane)
    hit_point = valid_pts[closest_point_idx]
    return True, hit_point


''' ------------------------------- Rviz 3D visualization ------------------------------------ '''


def draw_3d_pointing_ray(arm_3_joints_xyz, cam_pose,
                         extend_meters_3d=3.0,
                         ):
    # Draw 3d.
    p1, p2 = get_joints_for_pointing(arm_3_joints_xyz)
    p3 = p1 + extend_meters_3d * (p2 - p1) / np.linalg.norm(p2 - p1)
    p1_w = cam_to_world(p1, cam_pose)
    p2_w = cam_to_world(p2, cam_pose)
    p3_w = cam_to_world(p3, cam_pose)
    RvizMarker.draw_link(
        VIZ_ID_RAY, p1_w, p3_w, _color='r')


def delete_pointing_ray():
    RvizMarker.delete_marker(VIZ_ID_RAY)


def draw_3d_hit_point(xyz, cam_pose):
    RvizMarker.draw_dot(
        VIZ_ID_HIT_POINT,
        cam_to_world(xyz, cam_pose),
        _color='r',
        _size=0.3)


def delete_hit_point():
    RvizMarker.delete_marker(VIZ_ID_HIT_POINT)


''' ------------------------------- 2D image drawer ------------------------------------- '''


def draw_2d_hit_point(xyz_hit, intrin_mat, img_disp,
                      xyz_shoulder=None,
                      color=[0, 0, 255]):
    xy1 = cam2pixel(xyz_hit, intrin_mat)
    cv2.circle(img_disp, xy1,
               radius=DOT_RADIUS_2D_HIT_POINT, color=color, thickness=-1)
    if xyz_shoulder is not None:  # Draw a line between the shoulder and hit_point.
        xy0 = cam2pixel(xyz_shoulder, intrin_mat)
        cv2.line(img_disp, xy0, xy1, color=color, thickness=LINE_WIDTH_2D_RAY)
    hit_point_2d = xy1
    return hit_point_2d


def draw_2d_pointing_ray(arm_3_joints_xyz, cam_pose,
                         intrin_mat, img_disp,
                         extend_meters_2d=0.5,
                         color=[0, 0, 255],  # red
                         ):
    '''
    Usage:
        If there is no hit point, use this function to draw a short ray.
        Else, use `draw_2d_hit_point` to draw both hit point and the ray.
    The ray starts from shoulder and ends at `extend_meters_2d` meters in front of the wrist.
    '''
    p1, p2 = get_joints_for_pointing(arm_3_joints_xyz)
    p3 = p2 + extend_meters_2d * (p2 - p1) / np.linalg.norm(p2 - p1)
    cv2.line(img_disp,
             cam2pixel(p1, intrin_mat),
             cam2pixel(p3, intrin_mat),
             color=color, thickness=LINE_WIDTH_2D_RAY)


''' ------------------------------- Main ------------------------------------- '''


def main(args):

    # -- Data reader.
    if args.data_source == "disk":
        data_reader = DataReader_DISK(args)
    else:
        data_reader = DataReader_ROS(args)
    ith_image = 0
    total_images = data_reader.total_images()

    # -- Result publisher.
    pub_res_img = ColorImagePublisher(TOPIC_RES_IMAGE)
    if not os.path.exists(DST_RES_IMAGE_VIZ):
        os.makedirs(DST_RES_IMAGE_VIZ)

    # -- Detector.
    # Openpose.
    skeleton_detector = OpenposeDetector(
        {"hand": args.detect_hand})
    # YOLOv3.
    if args.detect_object:
        yolo_sub = YoloSubscriber(YOLO_TOPIC_NAME, OBJECT_CLASSES)

    # -- Camera pose (For Rviz visualization).
    cam_pose, cam_pose_pub = set_default_params()
    if args.is_using_realsense:  # Change coordinate.
        R, p = get_Rp_from_T(cam_pose)
        R = roty(math.pi/2).dot(rotz(-math.pi/2)).dot(R)
        cam_pose = form_T(R, p)

    # -- Loop: read, detect, draw.
    prev_humans = []
    while not rospy.is_shutdown() and ith_image < total_images:

        # -- Read data
        print("============================================")
        rospy.loginfo("Reading {}/{}th color/depth images...".format(
            ith_image+1, total_images))
        rgbd = data_reader.read_next_data()
        ith_image += 1
        t0 = time.time()

        pcl_xyz = rgbd.create_point_cloud(depth_max=4.0)  # The point cloud.
        color = rgbd.get_color_image()
        intrin_mat = rgbd.intrinsic_matrix()

        # Set camera pose only for visualizing 3d joints in Rviz.
        # (I tranform the ROS Markers' positions from camera frame to world frame,
        # and then publish them to ROS topic.)
        rgbd.set_camera_pose(cam_pose)

        # -- Detect joints.
        rospy.loginfo("  Detecting human skeletons by Openpose...")
        body_joints, hand_joints = skeleton_detector.detect(
            rgbd.color_image(), is_return_joints=True)
        img_disp = skeleton_detector.get_img_viz()
        N_people = len(body_joints)

        # -- Delete previous drawings in Rviz.
        for human in prev_humans:
            # TODO: Here is a bug. If I do this deletion of markers
            # after drawing new markders,
            # The deletion sometimes don't work. I don't know why.
            human.delete_rviz()

        # -- Process the 1st human's pointing direction.
        # Only process one human!!! (TODO: Remove this.)
        N_people = 1 if N_people >= 1 else 0
        humans = []
        is_pointing, is_hit = False, False
        rospy.loginfo("  Start 3D pointing detection...")
        for i in range(N_people):

            # Construct class Human.
            # Way 1: Use accumulated ID.
            human = Human(rgbd, body_joints[i], hand_joints[i])
            # Way 2: Use fixed ID. This is also fine.
            # human = Human(rgbd, body_joints[i], hand_joints[i], id=VIZ_HUMAN_ID)
            human.draw_rviz()

            rospy.loginfo("  Process {}/{}th person with id={}.".format(
                i+1, N_people, human._id))

            # -- 3D pointing detection.
            is_arm_exist, arm_3_joints_xyz = human.get_right_arm()
            if is_arm_exist:
                rospy.loginfo("    Right arm exists.")
                is_pointing = is_arm_stretched(arm_3_joints_xyz)
                # If the arm is fully stretched,
                # then the person is doing a "pointing" action.
                if is_pointing:
                    rospy.loginfo("    The person is pointing.")
                    draw_3d_pointing_ray(
                        arm_3_joints_xyz, cam_pose)
                    is_hit, xyz_hit = get_3d_ray_hit_point(
                        arm_3_joints_xyz, pcl_xyz)
                    # is_hit: True if there a 3D pixel
                    #   that is very close to the ray of pointing.
                    if is_hit:
                        rospy.loginfo("    Ray of pointing hits a 3D pixel.")
                        hit_point_2d = draw_2d_hit_point(
                            xyz_hit, intrin_mat, img_disp,
                            xyz_shoulder=arm_3_joints_xyz[0])
                        draw_3d_hit_point(xyz_hit, cam_pose)

                        # Check if `hit_point_2d` is in any of
                        #   the objects' bounding boxes detected by YOLOv3.
                        if args.detect_object:
                            bboxes = yolo_sub.get_bboxes()
                            box_id = in_which_box(hit_point_2d, bboxes)
                            if box_id >= 0:
                                rospy.loginfo(
                                    "    An object is being pointed.")
                                draw_box(img_disp, bboxes[box_id],
                                         color=[0, 0, 255], thickness=4)
                    else:
                        # If the ray of pointing doesn't hit any 3D pixel,
                        # we draw a short ray to indicate the person is pointing.
                        draw_2d_pointing_ray(arm_3_joints_xyz, cam_pose,
                                             intrin_mat, img_disp,
                                             extend_meters_2d=1.0)

            humans.append(human)

        # -- Delete markers.
        if not is_pointing:
            delete_pointing_ray()
        if not is_hit:
            delete_hit_point()

        # -- Print results.
        rospy.loginfo("Total time = {} seconds.".format(time.time()-t0))
        pub_res_img.publish(img_disp)
        cv2.imwrite(DST_RES_IMAGE_VIZ +
                    "/{:05d}.png".format(ith_image), img_disp)

        # -- Keep update camera pose for rviz visualization.
        cam_pose_pub.publish()
        prev_humans = humans

        # -- Reset data.
        if args.data_source == "disk" and ith_image == total_images:
            data_reader = DataReader_DISK(args)
            ith_image = 0

    # -- Clean up.
    for human in humans:
        human.delete_rviz()


if __name__ == '__main__':
    node_name = "detect_and_draw_joints"
    rospy.init_node(node_name)
    rospy.sleep(0.1)
    args = parse_command_line_arguments()
    main(args)
    rospy.logwarn("Node `{}` stops.".format(node_name))
