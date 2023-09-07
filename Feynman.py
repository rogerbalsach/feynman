#!/usr/bin/env python
# coding: utf-8

"""
Created on 2019

@author: Roger Balsach
"""

from itertools import product

import numpy as np
from numpy import allclose, arccos, arctan2, array, append, conjugate, cos,\
    copy, diag, eye, isinf, isnan, ndarray, sin, sqrt, trace, pi, prod, where
from numpy.linalg import multi_dot
from numpy.random import choice, randn, random, uniform
from scipy.constants import physical_constants

m_e = physical_constants['electron mass energy equivalent in MeV'][0]
m_mu = physical_constants['muon mass energy equivalent in MeV'][0]

m_W = 80379
m_Z = 91187.6


class FeynmanError(Exception):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)


def _most_common_class(a, b):
    for clas in type(a).mro():
        if clas in type(b).mro():
            return clas


class Vector:
    def __init__(self, coord=None, *, real=True, name='', size=1):
        self.name = name
        self.coord = array(4*['r']) if coord is None else array(coord)
        p = copy(self.coord)  # Makes a copy of the components to work with.
        if 'r' in self.coord.tolist():
            mask = (self.coord == 'r')
            # Avoids the problem of having "str" when converting the components
            # to float.
            p[mask] = 0
            # Tells the array to accept float components and not convert them
            # into int.
            if real:
                p = p.astype(np.float128)
                p[mask] = size*randn(mask.sum())
            else:
                p = p.astype(np.complex128)
                random = size*randn(2, mask.sum())
                p[mask] = random[0] + 1j*random[1]
        else:
            p = p.astype(np.float128) if real else p.astype(np.complex128)
        self.mu = p  # Returns the contravariant coordinates of the vector.

    def dot(self, other):  # Returns the dot product with another 4-vector.
        p = self.mu
        q = other.mu if isinstance(other, Vector) else other
        return dot(p, q)

    def rotate(self, z1=0, y2=0, z3=0):
        self.mu = Rotation(self.mu, z1, y2, z3)

    @property
    def _mu(self):  # Returns the covariant coordinates of the vector.
        p = self.mu
        return array([p[0]]+[-p[i] for i in range(1, 4)])

    @property
    def v2(self):
        return self.dot(self)

    @property
    def s(self):  # Returns the matrix 'v-slash'.
        return self.dot(Gamma_matrices)

    def conj(self, **kwargs):
        return Vector(conjugate(self.mu, **kwargs), real=False)

    def __eq__(self, other):
        inst = type(self)
        if not isinstance(other, inst):
            return False
        return allclose(self.mu, other.mu)

    def __add__(self, other, name=None):
        if isinstance(other, Vector):
            p = self.mu + other.mu
            if name is None:
                name = f'{self.name} + {other.name}'
            clas = _most_common_class(self, other)
            return clas(coord=p, name=name)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, complex, np.number)):
            p = other * self.mu
            name = f'{other}*{self.name}'
            return type(self)(coord=p, name=name)
        elif isinstance(other, (Vector, list)):
            return self.dot(other)
        return NotImplemented

    def __neg__(self):
        return -1*self

    def __sub__(self, other, name=None):
        if isinstance(other, Vector):
            return self + (-1)*other
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex, np.number)):
            return self*other
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex, np.number)):
            return self*(1/other)
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, int):
            if other % 2 == 0:
                return (self*self)**(other/2)
            else:
                return (self*self)**(other//2)*self
        return NotImplemented

    def __getitem__(self, i):
        # if isinstance(i, slice):
        #     if i[0] < 0:
        #         raise FeynmanError('Index of a 4 vector must be positive.')
        # if i < 0:
        #     raise FeynmanError('Index of a 4 vector must be positive.')
        return self.mu[i]

    def __call__(self, i):
        if i >= 0:
            return self.mu[i]
        else:
            return self._mu[-i]

    def __repr__(self):  # Changes the representation of the momentum.
        p = self.mu
        return f"Vector(coord=np.{repr(p)}, name='{self.name}')"

    def __str__(self):
        return f"'{self.name}' 4-Vector, a={self.mu}, a^2={self.v2:.2f}"


class Momentum(Vector):  # Defines a class of momentum-like 4-vectors
    def __init__(self, M=None, coord=None, *, name=None, size=1):
        super().__init__(coord=coord, name=name, size=size)
        if 'r' in self.coord.tolist():
            mask = (self.coord == 'r')
            r_idxs = where(mask)[0]
        else:
            r_idxs = array([0])
        p = self.mu
        if M is not None:  # Fixes the mass.
            if r_idxs[0] != 0:  # If the E is fixed.
                # Choose one random component.
                size = 0
                for mu in where(coord != 'r')[0]:
                    size += eta(mu, mu) * p[mu]**2
                if size < M**2:
                    print('coords incompatible with the given mass.')
                    p[0] = sqrt(M**2 + p[1]**2 + p[2]**2 + p[3]**2)
                else:
                    coord_to_change = choice(r_idxs)
                    fixed_coord = {1, 2, 3} - {coord_to_change}
                    fixed_mass, sign = p[0]**2, choice((1, -1))
                    for i in fixed_coord:
                        fixed_mass -= p[i]**2
                    while fixed_mass - M**2 < 0:
                        p[mask] = size * randn(mask.sum())
                        fixed_mass = p[0]**2
                        for i in fixed_coord:
                            fixed_mass -= p[i]**2
                        size /= 2
                    p[coord_to_change] = sign * sqrt(fixed_mass-M**2)
            else:
                p[0] = sqrt(M**2 + p[1]**2 + p[2]**2 + p[3]**2)

    # Returns the square of the vector (the mass square if it's On-Shell).
    @property
    def m2(self):
        return self.v2

    @property
    def Id(self):
        return self.mu[:, None, None] * Identity[None, ...]

    # Return the u spinor of momentum p and canonical polarization r.
    def u(self, r):
        p = self.mu
        if self.m2 < -1e-10:
            msg = 'Polarization is only defined for On-Shell particles!'
            raise FeynmanError(msg)
        m = sqrt(abs(self.m2))
        if r == 1:
            A = array([[1], [0], [p[3]/(p[0]+m)], [(p[1]+1j*p[2])/(p[0]+m)]])
            return sqrt(p[0] + m) * A
        if r == 2:
            A = array([[0], [1], [(p[1]-1j*p[2])/(p[0]+m)], [-p[3]/(p[0]+m)]])
            return sqrt(p[0] + m) * A

    def u_(self, r):
        return (self.u(r).T).conj() @ gamma(0)  # Return the adjoint of u.

    # Return the v spinor of momentum p and canonical polarization r.
    def v(self, r):
        p = self.mu
        if self.m2 < -1e-10:
            msg = 'Polarization is only defined for On-Shell particles!'
            raise FeynmanError(msg)
        m = sqrt(abs(self.m2))
        if r == 1:
            A = array([[(p[1]-1j*p[2])/(p[0]+m)], [-p[3]/(p[0]+m)], [0], [1]])
            return sqrt(p[0] + m) * A
        if r == 2:
            A = -array([[p[3]/(p[0]+m)], [(p[1]+1j*p[2])/(p[0]+m)], [1], [0]])
            return sqrt(p[0] + m) * A

    def v_(self, r):
        return (self.v(r).T).conj() @ gamma(0)  # Return the adjoint of v.

    def e(self, r):
        _, theta, phi = to_spherical(self.mu)
        if self.m2 < -1e-10:
            msg = 'Polarization is only defined for On-Shell particles!'
            raise FeynmanError(msg)
        m = sqrt(abs(self.m2))
        if r == 1:
            e = array([0, 1, 0, 0])
        if r == 2:
            e = array([0, 0, 1, 0])
        if r == 3 and m > 0:
            E = self(0)
            k = sqrt(E**2-m**2)
            e = array([k, 0, 0, E])/m
        return Vector(coord=Rotation(e, 0, theta, phi), real=False)

    def e_(self, r):
        p = self.e(r)
        return Vector(coord=array(p._mu), real=False)

    def __repr__(self):  # Changes the representation of the momentum.
        p = self.mu
        return f"Momentum(coord=np.{repr(p)}, name='{self.name}')"

    def __str__(self):
        return f"'{self.name}' 4-Momentum, p={self.mu}, p^2={self.m2:.2f}"


def gamma(i):  # gamma matrices (Dirac representation)
    if i == 0:
        return diag((1, 1, -1, -1))
    if i == 1:
        return array([[0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [0, -1, 0, 0],
                      [-1, 0, 0, 0]])
    if i == 2:
        return array([[0, 0, 0, -1j],
                      [0, 0, 1j, 0],
                      [0, 1j, 0, 0],
                      [-1j, 0, 0, 0]])
    if i == 3:
        return array([[0, 0, 1, 0],
                      [0, 0, 0, -1],
                      [-1, 0, 0, 0],
                      [0, 1, 0, 0]])
    if i == 5:
        return 1j * gamma(0) @ gamma(1) @ gamma(2) @ gamma(3)


def dot(a, b):
    '''
    Dot product of two vectors (with Minkowski metric).

    Parameters
    ----------
    a : List
        first 4-vector of the product.
    b : List
        second 4-vector of the product.

    Returns
    -------
    float
        Dot product usign Minkowski metric.

    '''
    return a[0]*b[0] - a[1]*b[1]-a[2]*b[2]-a[3]*b[3]


def square(a):
    return dot(a, a)


def delta(a, b):
    return 1 if a == b else 0


def eta(a, b):
    '''
    Defines the Minkowski metric with signature (+---):

    Parameters
    ----------
    a : int
        Integer between 0 and 3.
    b : int
        Integer between 0 and 3.

    Returns
    -------
    int
        1 if a=b=0, -1 if a=b!=0, 0 if a!=b.

    '''
    if a == -b:
        return 1
    elif a != b:
        return 0
    elif a == 0:
        return 1
    return -1


def comm(A, B):
    '''
    Commutator [A,B]

    Parameters
    ----------
    A : ndarray
        Left matrix.
    B : ndarray
        Right matrix.

    Returns
    -------
    ndarray
        Commutator of A and B.

    '''
    return A @ B - B @ A


def eps_(a, b, c, d):
    '''
    Defines the covariant levi-civita symbol in 4D.

    Parameters
    ----------
    a : int
        Integer between 0 and 3.
    b : int
        Integer between 0 and 3.
    c : int
        Integer between 0 and 3.
    d : int
        Integer between 0 and 3.

    Returns
    -------
    int
        1 if (a,b,c,d) is an even permutation of (0,1,2,3),
        -1 if (a,b,c,d) is an odd permutation of (0,1,2,3),
        0 if (a,b,c,d) is not a permutation of (0,1,2,3).

    '''
    if {0, 1, 2, 3} - {a, b, c, d}:
        return 0
    if a == 0 and b == 1 and c == 2 and d == 3:
        return 1
    elif b < a:
        return -eps_(b, a, c, d)
    elif c < b:
        return -eps_(a, c, b, d)
    elif d < c:
        return -eps_(a, b, d, c)
    assert False, f"this shouldn't run: {a=}, {b=}, {c=}, {d=}"


def eps(a, b, c, d):
    '''
    Returns the contravariant levi-civita symbol in 4D.

    Parameters
    ----------
    a : int
        Integer between 0 and 3.
    b : int
        Integer between 0 and 3.
    c : int
        Integer between 0 and 3.
    d : int
        Integer between 0 and 3.

    Returns
    -------
    int
        1 if (a,b,c,d) is an odd permutation of (0,1,2,3),
        -1 if (a,b,c,d) is an even permutation of (0,1,2,3),
        0 if (a,b,c,d) is not a permutation of (0,1,2,3).

    '''
    sgn = -1
    if a < 0:
        a *= -1
        sgn *= -1
    if b < 0:
        b *= -1
        sgn *= -1
    if c < 0:
        c *= -1
        sgn *= -1
    if d < 0:
        d *= -1
        sgn *= -1
    return sgn * eps_(a, b, c, d)


def eps_contraced(a, b, c, d):
    '''
    Returns the contraction of the 4-vectors a, b, c and d with the covariant
    levi-civita symbol.

    Parameters
    ----------
    a : List
        First 4-vector.
    b : List
        Second 4-vector.
    c : List
        Third 4-vector.
    d : List
        Fourth 4-vector.

    Returns
    -------
    r : float
        eps_(mu nu lambda sigma)*a^mu*b^nu*c^lambda*d^sigma.

    '''
    contracted = [a, b, c, d]
    vectors = 4*[Vector(coord=(1, -1, -1, -1))]
    if isinstance(a, Vector):
        contracted[0] = 9
        vectors[0] = a
    if isinstance(b, Vector):
        contracted[1] = 9
        vectors[1] = b
    if isinstance(c, Vector):
        contracted[2] = 9
        vectors[2] = c
    if isinstance(d, Vector):
        contracted[3] = 9
        vectors[3] = d
    contracted = np.asarray(contracted)
    if not sum(contracted == 9):
        return eps(a, b, c, d)
    a, b, c, d = vectors
    Idx = contracted.copy()
    res = 0
    for idx in Lorentz_Idx(sum(contracted == 9)):
        Idx[contracted == 9] = idx
        mu, nu, lamda, sigma = Idx
        res += (eps(mu, nu, lamda, sigma)
                * a(-abs(mu))*b(-abs(nu))*c(-abs(lamda))*d(-abs(sigma)))
    return res


def comprv(A, B, print_arg=False, tol=2*(1e-10,)):
    '''
    Compare the matrices A and B, element by element.
    Returns True if they are equal and False if they are not.

    The two values are considered equal if the absolute or the relative error
    is smaller than 1E-10.

    Parameters
    ----------
    A : number or array
        A number or array to be compared with B.
    B : number or array
        A number or array to be compared with A.
    print_arg : bool, optional
        If False (default) the function will only return the result.
        If True the function will also print the values of A, B with the
        corresponding errors and the result.
    e : tuple
        You can change the maximum errors for the function to return True.
        First component sets the maximum relative error. Default 1E-10.
        Second component sets the maximum absolute error. Default 1E-10.
    '''
    ab = abs(A - B)  # Computes the absolute error.
    try:  # Computes the relative error of B, raises an error if A=0.
        rel = abs((A - B) / A)
        if isnan(rel).any() or isinf(rel).any():
            raise ZeroDivisionError
    # Computes the relative error of A, raises an error if B=0:
    except ZeroDivisionError:
        try:
            rel = abs((A - B) / B)
            if isnan(rel).any() or isinf(rel).any():
                raise ZeroDivisionError
        # If A and B are zero, sets the relative error to 0:
        except ZeroDivisionError:
            if isinstance(rel, ndarray):
                rel[(isnan(rel) + isinf(rel))] = 0
            else:
                rel = 0
    # Bool that tells if the absolute and relative errors are within the
    # accepted values:
    res = ((rel < tol[0]) | (ab < tol[1]))
    if print_arg:
        print(A)
        print(B)
        print(ab, rel, res)
    return res


def average(M, Si, Sf):
    breakpoint()
    s = [1]*len(Si+Sf)
    S = append(Si, Sf)
    particle = 0
    r = M(s)
    while True:
        try:
            if s[particle] < S[particle]:
                s[particle] += 1
                r += M(s)
                particle = 0
            else:
                s[particle] = 1
                particle += 1
        except IndexError:
            return r/prod(Si)


def _to_spherical(x):
    r, theta = sqrt((x**2).sum()), 0
    if r != 0:
        x = x/r
        theta = arccos(x[2])
    phi = arctan2(x[1], x[0])
    return r, theta, phi


def to_spherical(x):
    '''
    x can be a Vector object, a 3-vector or a 4-vector.

    Returns the 3-vector in spherical coordinates.
    '''
    if isinstance(x, Vector):
        return _to_spherical(x[-3:])
    elif len(x) == 3:
        return _to_spherical(array(x))
    elif len(x) == 4:
        _, *x = x
        return _to_spherical(array(x))
    assert False


def _Rotation(v, z1, y2, z3):
    R1 = array([[cos(z1), -sin(z1), 0], [sin(z1), cos(z1), 0], [0, 0, 1]])
    R2 = array([[cos(y2), 0, sin(y2)], [0, 1, 0], [-sin(y2), 0, cos(y2)]])
    R3 = array([[cos(z3), -sin(z3), 0], [sin(z3), cos(z3), 0], [0, 0, 1]])
    if v is None:
        return multi_dot((R3, R2, R1))
    return append(v[:-3], multi_dot((R3, R2, R1, v[-3:])))


def Rotation(v=None, z1=0, y2=0, z3=0):
    """
    Parameters
    ----------
    v : tuple or Vector
        tuple containing several Vector objects to be rotated.
    z1 : float
        Angle for the first rotation (z-axis).
    y2 : float
        Angle for the second rotation (y-axis).
    z3 : float
        Angle for the first rotation (z-axis).

    Returns
    -------
    TYPE
        Rotated vector in the same format as input. (List are converted
        to tuples)
    """
    if isinstance(v, (tuple, list)):
        return tuple(Rotation(p, z1, y2, z3) for p in v)
    elif isinstance(v, Vector):
        return type(v)(coord=_Rotation(v, z1, y2, z3))
    elif not isinstance(v, ndarray):
        raise FeynmanError('Incorrect "v" variable.')
    return _Rotation(v, z1, y2, z3)


def zBoost(x, b, g):
    B = array([[g, 0, 0, -b*g], [0, 1, 0, 0], [0, 0, 1, 0], [-b*g, 0, 0, g]])
    if isinstance(x, (tuple, list)):
        return tuple(B@p for p in x)
    return B @ x


def Boost(x, b, y=None):
    g = 1 / sqrt(1 - b**2)
    if y is None:
        return zBoost(x, b, g)
    _, theta, phi = to_spherical(y)
    x1 = Rotation(x, -phi, -theta, 0)
    x2 = zBoost(x1, b, g)
    return Rotation(x2, 0, theta, phi)


def Källén(a, b, c):
    return a**2 + b**2 + c**2 - 2*a*b - 2*a*c - 2*b*c


def Generate_decay_1x2(M, RF='CM', Names=None):
    '''

    Parameters
    ----------
    M : List tuple or dict.
        Values for the three masses (the mass of the mother must be the first).
        If M is a dict, keys will be taken as the names and the values will be
        taken as the masses
    RF : 'CM', optional
        Choses the reference frame, only 'CM' allowed by now. The default is
        'CM'.
    Names : List of strings, optional
        Name for the three 4-momenta. The default is None.

    Returns
    -------
    p1 : 4-Momentum
        4-Momentum of the mother particle.
    p2, p3 : 4-Momentum
        4-Momentum of the final particles.

    '''
    if not Names:
        Names = 3 * ['']
    if RF == 'CM':
        if isinstance(M, dict):
            Names = list(M.keys())
            M = list(M.values())
        m1, m2, m3 = M
        if m1 < m2 + m3:
            print('This decay is not kinematically allowed!')
            return None
        K = Källén(m1**2, m2**2, m3**2)
        p = sqrt(float(K))/(2*m1)
        theta, phi = arccos(2*random()-1), 2*pi*random()
        p1 = Momentum(coord=array([m1, 0, 0, 0]), name=Names[0])
        p2 = Momentum(m2, array(['r', 0, 0, p]), Names[1])
        p3 = Momentum(m3, p2._mu, Names[2])
        p1.mu, p2.mu, p3.mu = Rotation((p1, p2, p3), 0, theta, phi)
        return p1, p2, p3
    elif isinstance(RF, Momentum):
        pass
    elif RF is None:
        pass


# TODO: Create an equivalent function Generate_scattering_2x3
def Generate_scattering_2x2(M, s=None, t=None, u=None, RF='CM', Names=4*['']):
    if isinstance(M, (float, int)):
        M = 4 * [M, ]
    elif isinstance(M, dict):
        Names = list(M.keys())
        M = list(M.values())
    m1, m2, m3, m4 = M
    if s is None:
        try:
            s = m1**2 + m2**2 + m3**2 + m4**2 - t - u
        except TypeError:
            s = uniform(1, 10) * max((m1+m2)**2, (m3+m4)**2, 1)
    if s < max((m1+m2)**2, (m3+m4)**2):
        print(f's need to be greater than {max((m1+m2)**2, (m3+m4)**2)}.')
        return None
    E1, E3 = (s + m1**2 - m2**2)/(2*sqrt(s)), (s + m3**2 - m4**2)/(2*sqrt(s))
    p, q = sqrt(E1**2-m1**2), sqrt(E3**2-m3**2)
    if t is None:
        try:
            t = m1**2 + m2**2 + m3**2 + m4**2 - s - u
        except TypeError:
            t = uniform(m1**2+m3**2-2*E1*E3-2*p*q, m1**2+m3**2-2*E1*E3+2*p*q)
    cq = (t-m1**2-m3**2+2*E1*E3)/(2*p*q)
    if abs(cq) > 1:
        msg = f't need to be between {m1**2+m3**2-2*E1*E3-2*p*q:.2f} and'\
            + '{m1**2+m3**2-2*E1*E3+2*p*q:.2f}.'
        raise ValueError(msg)
    if RF == 'CM':
        sq = sqrt(1-cq**2)
        p1 = Momentum(m1, array([E1, 'r', 'r', 'r']), size=p, name=Names[0])
        p2 = Momentum(m2, p1._mu, name=Names[1])
        _, theta, phi = to_spherical(p1.mu)
        coord = Rotation(q*array([sq, 0, cq]), 2*pi*random(), theta, phi)
        p3 = Momentum(m3, coord=['r', *coord], name=Names[2])
        p4 = Momentum(m4, p3._mu, name=Names[3])
        return p1, p2, p3, p4
    elif RF == 'Lab':
        P = Generate_scattering_2x2(M, s, t, Names=Names)
        p2 = P[1].mu
        b = sqrt(1-P[1].m2/p2[0]**2)
        _, theta, phi = to_spherical(P[0].mu)
        _P = Boost(P, b, p2)
        P[0].mu, P[1].mu, P[2].mu, P[3].mu = Rotation(_P, -phi, -theta, 0)
        return P


def _generate_scattering_2x3_00000(s, x):
    '''
    Computes the 4-momenta of the particles together with phase space factor.
    All particles must be massless. The result is given in the CM frame with p1
    in the z direction and p3 in the yz plane.


    Parameters
    ----------
    s : float
        s = (p1 + p2)**2. s must be a positive number
    s34 : float
        s34 = (p3 + p4)**2. s34 must be a positive number smaller than s.
    cqCMS : float
        cCMS is the cosine of the angle between p1 and p5 in the CM frame.
        Must be between -1 and 1
    PhiRest : float
        phi is the azimutal angle between p3 and p34 in the 34 rest frame.
    cqR : float
        cR is the cosine of the angle between p3 and p34 in the 34 rest frame.
        Must be between -1 and 1.

    Returns
    -------
    (numpy.ndarray, float)
        Matrix with 5 4-momenta (in columns, i.e p_i = P[:, i]) and the phase
        space integration factor with a factor 256pi⁴.

    '''
    s34, cqCMS, PhiRest, cqR = x

    E, E34 = sqrt(s), sqrt(s34)
    p34CMS = E*(1-s34/s)/2
    sqCMS, sqR = sqrt(1-cqCMS**2), sqrt(1-cqR**2)

    sqRcpR = sqR*cos(PhiRest)

    # auxiliar vector
    # a = (1 + p34CMS*cqR/(E34CMS + E34)) * p34CMS
    # a = (p34CMS + (E-E34)**2*cqR/(2*E))
    a = (E-E34)**2*cqR/(2*E)

    P = np.zeros((4, 5))  # P1, P2, P3, P4, P5
    P[0, 0] = E/2  # E1
    P[3, 0] = E/2  # p1z
    P[0, 1] = E/2  # E2
    P[3, 1] = -E/2  # p2z
    P[0, 2] = (E - p34CMS*(1-cqR))/2  # E3
    P[1, 2] = -E34*sqR*sin(PhiRest)/2  # p3x
    P[2, 2] = -(E34*(sqRcpR*cqCMS + sqCMS*cqR) + (a + p34CMS)*sqCMS)/2  # p3y
    P[3, 2] = -(E34*(cqCMS*cqR - sqRcpR*sqCMS) + (a + p34CMS)*cqCMS)/2  # p3z
    P[0, 4] = p34CMS  # E5
    P[2, 4] = p34CMS*sqCMS  # p5y
    P[3, 4] = p34CMS*cqCMS  # p5z

    P[0, 3] = (E - p34CMS*(1+cqR))/2  # E4
    P[1, 3] = E34*sqR*sin(PhiRest)/2  # p4x
    P[2, 3] = (E34*(sqRcpR*cqCMS + sqCMS*cqR) + (a - p34CMS)*sqCMS)/2  # p4y
    P[3, 3] = (E34*(cqCMS*cqR - sqRcpR*sqCMS) + (a - p34CMS)*cqCMS)/2  # p4z

    return P


def _generate_scattering_2x3_00110(s, x, m2, **kwargs):
    '''
    Computes the 4-momenta of the particles together with phase space factor.
    p1, p2, p5 must be massless. p3 and p4 must have the same mass. The result
    is given in the CM frame with p1 in the z direction and p3 in the yz plane.


    Parameters
    ----------
    s : float
        s = (p1 + p2)**2.
    x : tuple
        Tuple containing (s34, cCMS, phi, cR).
        s34 = (p3 + p4)**2
        cCMS is the cosine of the angle between p1 and p5.
        phi is the azimutal angle between p3 and p34.
        cR is the cosine of the angle between p3 and p34 in the 34 RF.
    M2 : float
        Squared mass of particles 3 and 4.

    Returns
    -------
    (numpy.ndarray, float)
        Matrix with 5 4-momenta (in columns, i.e p_i = P[:, i]) and the phase
        space integration factor with a factor 256pi⁴.

    '''
    s34, cqCMS, PhiRest, cqR = x

    E, E34 = sqrt(s), sqrt(s34)
    E34CMS = (s + s34)/(2*E)
    p34CMS = sqrt(E34CMS**2 - s34)
    sqCMS, sqR = sqrt(1-cqCMS**2), sqrt(1-cqR**2)

    E3Rest = E34/2
    p3Rest = sqrt(E3Rest**2 - m2)
    sqRcpR = sqR*cos(PhiRest)

    a = (p34CMS*p3Rest*cqR/(E34CMS + E34) + E3Rest) * p34CMS / E34

    P = np.zeros((4, 5))  # P1, P2, P3, P4, P5
    P[0, 0] = P[3, 0] = P[0, 1] = E/2  # E1, p1z, E2
    P[3, 1] = -E/2  # p2z
    P[0, 2] = (E34CMS*E3Rest+p34CMS*p3Rest*cqR)/E34  # E3
    P[1, 2] = -p3Rest*sqR*sin(PhiRest)  # p3x
    P[2, 2] = -p3Rest*(sqRcpR*cqCMS + sqCMS*cqR) - a*sqCMS  # p3y
    P[3, 2] = -p3Rest*(cqCMS*cqR - sqRcpR*sqCMS) - a*cqCMS  # p3z
    P[0, 4] = E - E34CMS  # E5
    P[2, 4] = p34CMS*sqCMS  # p5y
    P[3, 4] = p34CMS*cqCMS  # p5z

    P[:, 3] = P[:, 0] + P[:, 1] - P[:, 2] - P[:, 4]  # P4

    return P


def _generate_scattering_2x3_11220(s, x, M2, **kwargs):
    '''
    Computes the 4-momenta of the particles together with phase space factor.
    Initial particles (1 and 2) must have the same mass, particles 3 and 4 also
    must have the same mass, and particle 5 must be massless.
    The result is given in the CM frame with p1 in the z direction and p3 in
    the yz plane.


    Parameters
    ----------
    s : float
        s = (p1 + p2)**2.
    x : tuple
        Tuple containing (s34, cCMS, phi, cR).
        s34 = (p3 + p4)**2
        cCMS is the cosine of the angle between p1 and p5.
        phi is the azimutal angle between p3 and p34.
        cR is the cosine of the angle between p3 and p34 in the 34 RF.
    M2 : list, tuple, numpy.ndarray
        list containing m1² and m3².

    Returns
    -------
    (numpy.ndarray, float)
        Matrix with 5 4-momenta (in columns, i.e p_i = P[:, i]) and the phase
        space integration factor with a factor 256pi⁴.

    '''
    m, M = M2
    s34, cqCMS, PhiRest, cqR = x

    E, E34 = sqrt(s), sqrt(s34)
    E34CMS = (s + s34)/(2*E)
    p34CMS = sqrt(E34CMS**2 - s34)
    sqCMS, sqR = sqrt(1-cqCMS**2), sqrt(1-cqR**2)

    E3Rest = E34/2
    p3Rest = sqrt(E3Rest**2 - M)
    sqRcpR = sqR*cos(PhiRest)

    a = (p34CMS*p3Rest*cqR/(E34CMS + E34) + E3Rest) * p34CMS / E34

    E1 = E/2
    p1 = sqrt(E1**2 - m)
    P = np.zeros((4, 5))  # P1, P2, P3, P4, P5
    P[0, 0] = E1  # E1
    P[3, 0] = p1  # p1z
    P[0, 1] = E1  # E2
    P[3, 1] = -p1  # p2z
    P[0, 2] = (E34CMS*E3Rest+p34CMS*p3Rest*cqR)/E34  # E3
    P[1, 2] = -p3Rest*sqR*sin(PhiRest)  # p3x
    P[2, 2] = -p3Rest*(sqRcpR*cqCMS + sqCMS*cqR) - a*sqCMS  # p3y
    P[3, 2] = -p3Rest*(cqCMS*cqR - sqRcpR*sqCMS) - a*cqCMS  # p3z
    P[0, 4] = E - E34CMS  # E5
    P[2, 4] = p34CMS*sqCMS  # p5y
    P[3, 4] = p34CMS*cqCMS  # p5z

    P[:, 3] = P[:, 0] + P[:, 1] - P[:, 2] - P[:, 4]  # P4

    return P


def _generate_scattering_2x3_12345(s, x, M2, **kwargs):
    '''
    Computes the 4-momenta of the particles together with phase space factor.
    The result is given in the CM frame with p1 in the z direction and p3 in
    the yz plane.


    Parameters
    ----------
    s : float
        s = (p1 + p2)**2.
    x : tuple
        Tuple containing (s34, cCMS, phi, cR).
        s34 = (p3 + p4)**2
        cCMS is the cosine of the angle between p1 and p5.
        phi is the azimutal angle between p3 and p34.
        cR is the cosine of the angle between p3 and p34 in the 34 RF.
    M2 : list, tuple, numpy.ndarray
        list containing the squared masses of the 5 particles.

    Returns
    -------
    (numpy.ndarray, float)
        Matrix with 5 4-momenta (in columns, i.e p_i = P[:, i]) and the phase
        space integration factor with a factor 256pi⁴.

    '''
    m1, m2, m3, m4, m5 = M2
    s34, cqCMS, PhiRest, cqR = x

    E, E34 = sqrt(s), sqrt(s34)
    E34CMS = (s + s34 - m5)/(2*E)  # Energy of the 34 system in the cm frame
    p34CMS = sqrt(E34CMS**2 - s34)  # momentum of the 34 system in the cm frame
    sqCMS, sqR = sqrt(1-cqCMS**2), sqrt(1-cqR**2)

    E3Rest = (s34 + m3 - m4)/(2*E34)  # Energy of p4 in the 45 rest frame.
    p3Rest = sqrt(E3Rest**2 - m3)  # momentum of p3 in the 45 rest frame.
    sqRcpR = sqR*cos(PhiRest)

    a = (p34CMS*p3Rest*cqR/(E34CMS + E34) + E3Rest) * p34CMS / E34

    E1 = (s + m1 - m2)/(2*E)
    p1 = sqrt(E1**2 - m1)
    P = np.zeros((4, 5))  # P1, P2, P3, P4, P5
    P[0, 0] = E1  # E1
    P[3, 0] = p1  # p1z
    P[0, 1] = E - E1  # E2
    P[3, 1] = -p1  # p2z
    P[0, 2] = (E34CMS*E3Rest+p34CMS*p3Rest*cqR)/E34  # E3
    P[1, 2] = -p3Rest*sqR*sin(PhiRest)  # p3x
    P[2, 2] = -p3Rest*(sqRcpR*cqCMS + sqCMS*cqR) - a*sqCMS  # p3y
    P[3, 2] = -p3Rest*(cqCMS*cqR - sqRcpR*sqCMS) - a*cqCMS  # p3z
    P[0, 4] = E - E34CMS  # E5
    P[2, 4] = p34CMS*sqCMS  # p5y
    P[3, 4] = p34CMS*cqCMS  # p5z

    P[:, 3] = P[:, 0] + P[:, 1] - P[:, 2] - P[:, 4]  # P4

    return P


def Generate_scattering_2x3(s, x, M, **kwargs):
    '''
    Computes the 4-momenta of the particles together with phase space factor.
    The result is given in the CM frame with p1 in the z direction and p3 in
    the yz plane.


    Parameters
    ----------
    s : float
        s = (p1 + p2)**2.
    x : tuple
        Tuple containing (s34, cCMS, phi, cR).
        s34 = (p3 + p4)**2
        cCMS is the cosine of the angle between p1 and p5.
        phi is the azimuthal angle between p3 and p34 in the 34 RF.
        cR is the cosine of the angle between p3 and p34 in the 34 RF.
    M : list, tuple, numpy.ndarray
        list containing the masses of the 5 particles.

    Returns
    -------
    (numpy.ndarray, float)
        Matrix with 5 4-momenta (in columns, i.e p_i = P[:, i]) and the phase
        space integration factor.

    '''
    m1, m2, m3, m4, m5 = (m**2 for m in M)
    if not hasattr(x, '__getitem__'):
        raise ValueError(f'{x} must be indexable.')
    if not hasattr(x, '__len__'):
        raise ValueError(f'{x} must have a defined length.')
    if len(x) != 4:
        raise ValueError(f'{x} must have 4 elements.')
    E, E34 = sqrt(s), sqrt(x[0])
    if not E > M[2] + M[3] + M[4]:
        print(M, s)
        raise FeynmanError(
            's not valid. Must be greater than the final mass: '
            + f'{s = } > (Σm)^2 = {(M[2] + M[3] + M[4])**2}'
        )
    if not (M[2] + M[3] <= E34 <= E - M[4]):
        print(M, s, x[0])
        raise FeynmanError(
            's34 not valid. Must be between '
            + f'{(M[2] + M[3])**2} < s34={x[0]} < {(E - M[4])**2}'
        )
    if m1 == m2 and m3 == m4 and m5 == 0:
        if m1 == 0:
            if m3 == 0:
                return _generate_scattering_2x3_00000(s, x)
            else:
                return _generate_scattering_2x3_00110(s, x, m3)
        else:
            return _generate_scattering_2x3_11220(s, x, (m1, m3))
    else:
        return _generate_scattering_2x3_12345(s, x, (m1, m2, m3, m4, m5))


def Mandelstam(a, b=False, c=False, d=False):
    if isinstance(a, tuple):
        a, b, c, d = a
    elif not (bool(b) & bool(c) & bool(d)):
        print('4 4-Momenta are needed.')
        return None
    s = (a+b)**2
    t = (a-c)**2
    u = (a-d)**2
    return s, t, u


def Lorentz_Idx(n):
    if n == 0:
        return []
    elif n == 1:
        return range(4)
    return product(range(4), repeat=n)


def Polarizations(*pol):
    iters = []
    for n in pol:
        iters.append(range(1, n+1))
    if not iters:
        return []
    elif len(iters) == 1:
        return iters[0]
    return product(*iters)


Identity = eye(4)  # Defines the 4x4 identity matrix
Gamma_matrices = array([gamma(i) for i in range(4)])  # Gamma matrices
# Covariant Gamma matrices
Gamma_matrices_cov = array([gamma(0)] + [-gamma(i) for i in range(1, 4)])
# sigma matrices.
Sigma_matrices = (1j)/2*array([comm(gamma(i//4), gamma(i % 4))
                               for i in range(16)]).reshape((4, 4, 4, 4))

if __name__ == '__main__':
    a = Momentum()
    p1 = Momentum(0)
    e1 = p1.e(1)
    e2 = p1.e(2)
    e0 = Vector((1, 0, 0, 0))
    e3 = Vector((0, *p1[1:]/sqrt(p1[1]**2+p1[2]**2+p1[3]**2)))
