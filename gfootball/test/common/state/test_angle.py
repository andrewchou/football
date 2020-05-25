import unittest
from unittest import TestCase

import numpy as np

from gfootball.common.state.angle import relative_angle_bucket

class AngleTestCase(TestCase):
    def test_relative_angle_bucket(self):
        self.assertEqual(0, relative_angle_bucket(delta_position=np.array([1.0, -.1])))
        self.assertEqual(0, relative_angle_bucket(delta_position=np.array([1.0, 0.0])))
        self.assertEqual(0, relative_angle_bucket(delta_position=np.array([1.0, 0.1])))

        self.assertEqual(1, relative_angle_bucket(delta_position=np.array([1.0, 1.0])))
        self.assertEqual(2, relative_angle_bucket(delta_position=np.array([0.0, 1.0])))
        self.assertEqual(3, relative_angle_bucket(delta_position=np.array([-1.0, 1.0])))
        self.assertEqual(4, relative_angle_bucket(delta_position=np.array([-1.0, 0.0])))
        self.assertEqual(5, relative_angle_bucket(delta_position=np.array([-1.0, -1.0])))
        self.assertEqual(6, relative_angle_bucket(delta_position=np.array([0.0, -1.0])))
        self.assertEqual(7, relative_angle_bucket(delta_position=np.array([1.0, -1.0])))
        # No angle
        self.assertEqual(0, relative_angle_bucket(delta_position=np.array([0.0, 0.0])))

if __name__ == '__main__':
    unittest.main()
