import rospy
from visualization_msgs.msg import Marker

from .conversions import *


def create_cube_marker(frame, pose, scale, color, ns="", id=0):
    return create_marker(Marker.CUBE, frame, pose, scale, color, ns, id)


def create_line_list_marker(frame, pose, scale, color, lines, ns="", id=0):
    marker = create_marker(Marker.LINE_LIST, frame, pose, scale, color, ns, id)
    marker.points = [to_point_msg(point) for line in lines for point in line]
    return marker


def create_line_strip_marker(frame, pose, scale, color, points, ns="", id=0):
    marker = create_marker(Marker.LINE_STRIP, frame, pose, scale, color, ns, id)
    marker.points = [to_point_msg(point) for point in points]
    return marker


def create_sphere_marker(frame, pose, scale, color, ns="", id=0):
    return create_marker(Marker.SPHERE, frame, pose, scale, color, ns, id)


def create_sphere_list_marker(frame, pose, scale, color, centers, ns="", id=0):
    marker = create_marker(Marker.SPHERE_LIST, frame, pose, scale, color, ns, id)
    marker.points = [to_point_msg(center) for center in centers]
    return marker


def create_marker(type, frame, pose, scale, color, ns="", id=0):
    msg = Marker()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time()
    msg.ns = ns
    msg.id = id
    msg.type = type
    msg.action = Marker.ADD
    msg.pose = to_pose_msg(pose)
    msg.scale = to_vector3_msg(scale)
    msg.color = to_color_msg(color)
    return msg
