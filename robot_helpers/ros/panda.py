import actionlib
import numpy as np
import rospy

from franka_gripper.msg import *
from sensor_msgs.msg import JointState


class PandaArmClient:
    def __init__(self):
        rospy.Subscriber("/joint_states", JointState, self._joint_state_cb)
        rospy.wait_for_message("/joint_states", JointState)

    def get_state(self):
        q = np.asarray(self._joint_state_msg.position[:7])
        dq = np.asarray(self._joint_state_msg.velocity[:7])
        return q, dq

    def _joint_state_cb(self, msg):
        self._joint_state_msg = msg


class PandaGripperClient:
    def __init__(self, ns="/franka_gripper"):
        rospy.Subscriber("/joint_states", JointState, self._joint_state_cb)
        self._move_client = actionlib.SimpleActionClient(ns + "/move", MoveAction)
        self._grasp_client = actionlib.SimpleActionClient(ns + "/grasp", GraspAction)

    def move(self, width, speed=0.1):
        msg = MoveGoal(width, speed)
        self._move_client.send_goal(msg)
        self._move_client.wait_for_result()

    def grasp(self, width=0.0, e_inner=0.1, e_outer=0.1, speed=0.1, force=10.0):
        msg = GraspGoal(width, GraspEpsilon(e_inner, e_outer), speed, force)
        self._grasp_client.send_goal(msg)
        self._grasp_client.wait_for_result()

    def read(self):
        return self._joint_state_msg.position[7] + self._joint_state_msg.position[8]

    def _joint_state_cb(self, msg):
        self._joint_state_msg = msg
