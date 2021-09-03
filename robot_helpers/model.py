"""Compute kinematic properties of a robot using PyKDL."""

import numpy as np
import kdl_parser_py.urdf
import PyKDL as kdl
import urdf_parser_py.urdf

from .spatial import Rotation, Transform


class Model:
    def __init__(self, urdf, root_frame, tip_frame):
        self.urdf = urdf_parser_py.urdf.URDF.from_xml_file(urdf)
        _, tree = kdl_parser_py.urdf.treeFromUrdfModel(self.urdf)
        self.chain = tree.getChain(root_frame, tip_frame)
        self.fk_pos_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.fk_vel_solver = kdl.ChainFkSolverVel_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)

    def pose(self, frame, q):
        link_index = self._get_link_index(frame)
        jnt_array = joint_list_to_kdl(q)
        res = kdl.Frame()
        self.fk_pos_solver.JntToCart(jnt_array, res, link_index)
        return kdl_frame_to_transform(res)

    def velocities(self, frame, q, dq):
        link_index = self.link_names.index(frame)
        jnt_array_vel = kdl.JntArrayVel(joint_list_to_kdl(q), joint_list_to_kdl(dq))
        res = kdl.FrameVel()
        self.fk_vel_solver.JntToCart(jnt_array_vel, res, link_index)
        d = res.deriv()
        linear, angular = np.r_[d[0], d[1], d[2]], np.r_[d[3], d[4], d[5]]
        return linear, angular

    def jacobian(self, q):
        jnt_array = joint_list_to_kdl(q)
        J = kdl.Jacobian(self.chain.getNrOfJoints())
        self.jac_solver.JntToJac(jnt_array, J)
        return kdl_mat_to_array(J)

    def _get_link_index(self, frame):
        return self.urdf.links.index(self.urdf.link_map[frame])


def joint_list_to_kdl(q):
    jnt_array = kdl.JntArray(len(q))
    for i, q_i in enumerate(q):
        jnt_array[i] = q_i
    return jnt_array


def kdl_frame_to_transform(f):
    rotation = Rotation.from_matrix(
        np.array(
            [
                [f.M[0, 0], f.M[0, 1], f.M[0, 2]],
                [f.M[1, 0], f.M[1, 1], f.M[1, 2]],
                [f.M[2, 0], f.M[2, 1], f.M[2, 2]],
            ]
        )
    )
    translation = np.r_[f.p[0], f.p[1], f.p[2]]
    return Transform(rotation, translation)


def kdl_mat_to_array(m):
    # TODO map memory directly ?
    mat = np.zeros((m.rows(), m.columns()))
    for i in range(m.rows()):
        for j in range(m.columns()):
            mat[i, j] = m[i, j]
    return mat
