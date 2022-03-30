import copy

import numpy as np
from scipy.spatial.transform import Rotation


class Transform:
    def __init__(self, rotation, translation):
        self.rotation = copy.deepcopy(rotation)
        self.translation = np.asarray(translation, np.double).copy()

    @classmethod
    def from_rotation(cls, rotation):
        translation = np.zeros(3)
        return cls(rotation, translation)

    @classmethod
    def from_translation(cls, translation):
        rotation = Rotation.identity()
        return cls(rotation, translation)

    @classmethod
    def from_matrix(cls, m):
        rotation = Rotation.from_matrix(m[:3, :3])
        translation = m[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_list(cls, l):
        return cls(Rotation.from_quat(l[:4]), l[4:])

    def as_matrix(self):
        return np.vstack(
            (np.c_[self.rotation.as_matrix(), self.translation], [0.0, 0.0, 0.0, 1.0])
        )

    def to_list(self):
        return np.r_[self.rotation.as_quat(), self.translation]

    def __mul__(self, other):
        rotation = self.rotation * other.rotation
        translation = self.rotation.apply(other.translation) + self.translation
        return self.__class__(rotation, translation)

    def inv(self):
        rotation = self.rotation.inv()
        translation = -rotation.apply(self.translation)
        return self.__class__(rotation, translation)

    def apply(self, point):
        return self.rotation.apply(point) + self.translation

    @classmethod
    def identity(cls):
        rotation = Rotation.identity()
        translation = np.array([0.0, 0.0, 0.0])
        return cls(rotation, translation)

    class TClass:
        """
        Convenient way to create a pure translation.

        Transform.t_[x, y, z] is equivalent to Transform.from_translation(np.r_[x, y, z]).
        """

        def __getitem__(self, key):
            return Transform.from_translation(np.r_[key])

    t_ = TClass()


def look_at(eye, center, up):
    eye = np.asarray(eye)
    center = np.asarray(center)
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.asarray(up) / np.linalg.norm(up)
    up = np.cross(right, forward)
    m = np.eye(4, 4)
    m[:3, 0] = right
    m[:3, 1] = -up
    m[:3, 2] = forward
    m[:3, 3] = eye
    return Transform.from_matrix(m)
