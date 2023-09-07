# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 19:50:07 2021

@author: user
"""

import unittest

from numpy import allclose

from Feynman import *
from Feynman import array, eye, Generate_scattering_2x2, Momentum, ndarray,\
    randn, Rotation, to_spherical, uniform, pi, Vector
import Feynman as fey


class TestVector(unittest.TestCase):
    def test_name(self):
        v = Vector(name='vector')
        self.assertEqual(v.name, 'vector')
        self.assertTrue(all(v.coord == array(4*['r'])))

    def test_ran_coords(self):
        v = Vector(coord=array([-0.3,  'r', 0.6, 'r']))
        self.assertEqual(v.coord[1], 'r')
        self.assertEqual(v.coord[3], 'r')
        self.assertEqual(v[0], -0.3)
        self.assertEqual(v[2], 0.6)

    def test_coords(self):
        v = Vector(coord=array([-2.3,  1.1, -1.1, -0.3]))
        self.assertFalse('r' in v.coord.tolist())
        self.assertEqual(v[0], -2.3)
        self.assertEqual(v[1], 1.1)
        self.assertEqual(v[2], -1.1)
        self.assertEqual(v[3], -0.3)
        # self.assertEqual(round(v**2, 3), 2.78)
        self.assertAlmostEqual(v**2, 2.78, 8)

    def test_dot(self):
        v1 = Vector(coord=[0.9, -0.3, -1, -0.7])
        v2 = Vector(coord=[1.4, 0.8, 0.5, 0.6])
        self.assertEqual(v1.dot(v2), 2.42)
        self.assertEqual(v1*v2, 2.42)
        self.assertEqual(v1*[-1.1, -0.9,  2.7, -0.6], 1.02)

    def test_rotate(self):
        v1 = Vector(coord=[0.3, *(3*('r',))])
        v2 = Vector(coord=[0, 'r',  'r', -0.5])
        v3 = Vector(coord=[-0.5, -0.4,  0.8,  0.9])
        z1, y2, z3 = randn(3)
        v1.rotate(z1, y2, z3)
        v2.rotate(z1, 0, z3)
        v3.rotate(-0.3, 0.9, -1)
        self.assertEqual(v1[0], 0.3)
        self.assertEqual(v2[3], -0.5)
        self.assertEqual(v3, Vector(coord=array([-0.5,
                                                 1.07454842,
                                                 -0.04020721,
                                                 0.67359414])))

    def test_eq(self):
        v1 = Vector(coord=[0.1, -1.8,  0.6, -1.9])
        v2 = Vector(coord=[-0.1, -0.6, -0.6, -1])
        v3 = Vector(coord=array([0.1, -1.8,  0.6, -1.9]), name='name')
        self.assertTrue(v1 == v1)
        self.assertFalse(v1 == v2)
        self.assertTrue(v1 == v3)

    def test_add(self):
        v1 = Vector(coord=[-1.4, -0.2, -0.9, -1.4])
        v2 = Vector(coord=[0.5, -0.1,  0.6,  2])
        v3 = Vector(coord=[-0.9, -0.3,  -0.3,  0.6])
        self.assertEqual(v3, v1+v2)

    def test_repr(self):
        v = Vector(coord=[1.4, 0.7, -0.6, 0.1], name='vector')
        r = "Vector(coord=np.array([ 1.4,  0.7, -0.6,  0.1]), name='vector')"
        self.assertEqual(repr(v), r)


class TestMomentum(unittest.TestCase):
    def test_id(self):
        p = Momentum()
        pI = p.Id
        self.assertEqual(pI.shape, (4, 4, 4))
        self.assertTrue((pI[0] == p[0]*eye(4)).all())
        self.assertTrue((pI[1] == p[1]*eye(4)).all())
        self.assertTrue((pI[2] == p[2]*eye(4)).all())
        self.assertTrue((pI[3] == p[3]*eye(4)).all())

    def test_add(self):
        p1 = Momentum(coord=[0.1, -2.2, -1.6, 1.2])
        p2 = Momentum(coord=[-0.6, -2, 0.1, 0.4])
        p3 = Momentum(coord=[-0.5, -4.2, -1.5, 1.6])
        self.assertEqual(p3, p1+p2)

    def test_u(self):
        p = Momentum(0)
        self.assertEqual(p.u(1)[1], 0)
        self.assertEqual(p.u(2)[0], 0)

    def test_v_(self):
        p = Momentum(0)
        self.assertEqual(p.v_(1)[:, 2], 0)
        self.assertEqual(p.v_(2)[:, 3], 0)

    def test_e(self):
        p = Momentum(0)
        p.e(1)
        p.e(2)


class TestSpherical(unittest.TestCase):
    def test_vector(self):
        p = Vector(coord=array([-0.3,  0.1,  1.5,  1.2]))
        self.assertEqual(to_spherical(p), (1.9235384061671343,
                                           0.8971367295893576,
                                           1.5042281630190728))

    def test_3array(self):
        p = array([0.2,  0.8,  0.1])
        self.assertEqual(to_spherical(p), (0.8306623862918076,
                                           1.4501177736638868,
                                           1.3258176636680326))

    def test_4array(self):
        p = array([-1.1,  0,  0.4,  1])
        self.assertEqual(to_spherical(p), (1.077032961426901,
                                           0.38050637711236523,
                                           1.5707963267948966))


class TestRotation(unittest.TestCase):
    def test_type(self):
        z1, y2, z3 = uniform(0, 2*pi, 3)
        a = Vector()
        p = Momentum()
        b = array([0.1, 0.2, -0.1, 0])
        c = array([0.9, 1.3, 1.6])
        d = array([0.3, -0.6, -1.4, -0.2, 0.2, 1.1])
        self.assertIsInstance(Rotation(a, z1, y2, z3), Vector)
        self.assertIsInstance(Rotation(p, z1, y2, z3), Momentum)
        Rb = Rotation(b, z1, y2, z3)
        self.assertIsInstance(Rb, ndarray)
        self.assertEqual(len(Rb), 4)
        Rc = Rotation(c, z1, y2, z3)
        self.assertIsInstance(Rc, ndarray)
        self.assertEqual(len(Rc), 3)
        self.assertIsInstance(Rotation([a, b, c, p], z1, y2, z3), tuple)
        Rd = Rotation(d, z1, y2, z3)
        self.assertIsInstance(Rd, ndarray)
        self.assertEqual(len(Rd), 6)

    def test_id(self):
        R = Rotation(array([2, 0, 0, 2]), 0, 0, 0)
        self.assertTrue(allclose(R, array([2, 0, 0, 2])))

    def test_inv(self):
        a = Vector()
        _, q, p = to_spherical(a)
        R = Rotation(a, -p, -q, 0)
        self.assertAlmostEqual(R[1], 0, 14)
        self.assertAlmostEqual(R[2], 0, 14)
        # self.assertEqual(round(R[1], 15), 0)
        # self.assertEqual(round(R[2], 15), 0)


class TestScattering(unittest.TestCase):
    def test_dict(self):
        P = Generate_scattering_2x2({'p1': 10, 'p2': 10, 'p3': 1, 'p4': 6})
        self.assertIsInstance(P, tuple)
        self.assertEqual(len(P), 4)
        self.assertEqual(P[0].name, 'p1')
        self.assertAlmostEqual(P[0].m2, 100, 10)
        self.assertEqual(P[1].name, 'p2')
        self.assertAlmostEqual(P[1].m2, 100, 10)
        self.assertEqual(P[2].name, 'p3')
        self.assertAlmostEqual(P[2].m2, 1, 10)
        self.assertEqual(P[3].name, 'p4')
        self.assertAlmostEqual(P[3].m2, 36, 10)
        self.assertEqual(P[0]+P[1]-P[2]-P[3], Momentum(coord=[0, 0, 0, 0]))

    def test_t(self):
        with self.assertRaises(ValueError):
            Generate_scattering_2x2([8, 8, 4, 6], 1500, -1400)
        with self.assertRaises(ValueError):
            Generate_scattering_2x2([4, 2, 3, 3], 180, 1)
        with self.assertRaises(ValueError):
            Generate_scattering_2x2([2, 5, 5, 7], 510, 0)

    def test_massles(self):
        P = Generate_scattering_2x2((0, 0, 0, 0))
        self.assertAlmostEqual(P[0].m2, 0, 14)
        self.assertAlmostEqual(P[1].m2, 0, 14)
        self.assertAlmostEqual(P[2].m2, 0, 14)
        self.assertAlmostEqual(P[3].m2, 0, 14)


class TestConstants(unittest.TestCase):
    def test_masses(self):
        self.assertEqual(fey.m_mu, 105.6583755)
        self.assertEqual(fey.m_e, 0.51099895)


if __name__ == '__main__':
    unittest.main()
