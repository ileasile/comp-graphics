import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos
from itertools import combinations as combs
from scipy.spatial import ConvexHull

def rotate_mtx(ax = 0., ay=0., az=0.):
    x = np.array([
        [1, 0, 0, 0],
        [0, cos(ax), sin(ax), 0],
        [0, -sin(ax), cos(ax), 0],
        [0, 0, 0, 1],
    ])
    y = np.array([
        [cos(ay), 0, -sin(ay), 0],
        [0, 1, 0, 0],
        [sin(ay), 0, cos(ay), 0],
        [0, 0, 0, 1],
    ])
    z = np.array([
        [cos(az), sin(az), 0, 0],
        [-sin(az), cos(az), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    return np.matmul(z, np.matmul(y, x))

def perm_mtx(p):
    n = len(p)
    r = np.zeros((n, n), dtype=np.float32)
    for i, j in enumerate(p):
        r[i, j] = 1
    return r

def trans_mtx(s: str):
    r = np.identity(4)
    if 'x' in s:
        r[0, 0] = -1
    if 'y' in s:
        r[1, 1] = -1
    if 'z' in s:
        r[2, 2] = -1
    return r

def shift_mtx(s):
    r = np.identity(4)
    r[3, :3] = np.array(s)
    return r

t1 = np.diag([1/2., 1/3., 1., 1])
t2 = np.diag([1, 1, 1, 1/2.])
t3 = np.array([[1, -0.85, 0.25, 0],
               [-0.75, 1, 0.7, 0],
               [0.5, 1, 1, 0],
               [0, 0, 0, 1]])
t4 = rotate_mtx(ax = np.pi/2)
t5 = perm_mtx([1, 2, 0, 3])
t6 = trans_mtx('z')
t7 = shift_mtx([2, 1, 0])



class Shape:
    def __init__(self, surfaces):
        self.s = surfaces

    def transformed(self, transform):
        new_surfaces = []
        for t in self.s:
            new_surfaces.append(np.matmul(t, transform))
        return Shape(new_surfaces)

    def draw(self, ax, col, transparency=0.5):
        p = []
        for xc in self.s:
            for i in range(xc.shape[0]):
                for j in range(3):
                    xc[i, j] = xc[i, j] / xc[i, 3]
            p.append([tuple(xc[i, :3]) for i in range(xc.shape[0])])

        poly = Poly3DCollection(p)
        poly.set_color(list(col) + [transparency])
        poly.set_edgecolor('k')
        ax.add_collection3d(poly)


class Cube(Shape):
    def __init__(self, x):
        coords = [
            [0, 1, 2, 3],
            [0, 1, 5, 4],
            [0, 3, 7, 4],
            [6, 5, 1, 2],
            [6, 7, 3, 2],
            [6, 5, 4, 7],
        ]

        s = [x[c, :] for c in coords]

        super().__init__(s)


class CubeCut(Shape):
    def __init__(self, x):
        coords = [
            [0, 1, 2, 3, 4],
            [0, 1, 6, 5],
            [0, 5, 8, 4],
            [7, 8, 5, 6],
            [7, 9, 2, 1, 6],
            [7, 8, 4, 3, 9],
            [3, 2, 9]
        ]

        s = [x[c, :] for c in coords]

        super().__init__(s)

def draw_show_figs(figs, cols, liml=-4, limr=4):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_aspect('equal')
    ax.set_xlim(liml, limr)
    ax.set_ylim(liml, limr)
    ax.set_zlim(liml, limr)

    # ax.axhline(y=0, color='k')
    # ax.axvline(x=0, color='k')

    for i, f in enumerate(figs):
        f.draw(ax, cols[i])

    plt.show()

if __name__ == "__main__":
    plt.ioff()

    x = np.array([
        [0, 0, 1, 1],
        [2, 0, 1, 1],
        [2, 3, 1, 1],
        [0, 3, 1, 1],
        [0, 0, 0, 1],
        [2, 0, 0, 1],
        [2, 3, 0, 1],
        [0, 3, 0, 1]
    ], dtype=np.float32)

    y = np.array([
        [2, 1, 2, 1],
        [3, 1, 2, 1],
        [3, 1.5, 2, 1],
        [2.5, 2, 2, 1],
        [2, 2, 2, 1],
        [2, 1, 1, 1],
        [3, 1, 1, 1],
        [3, 2, 1, 1],
        [2, 2, 1, 1],
        [3, 2, 1.5, 1],
    ], dtype=np.float32)

    colors = [[1, 0, 0], [0, 0.5, 0], [0, 0, 0.7]]

    # ch = ConvexHull(x, qhull_options="Qu")

    trans_cube = [
        [t1, 0, 3],
        [t2, 0, 6],
        [t3, -1, 5],
        [t4, -1, 3],
        [t5, 0, 3],
        [t6, 0, 3],
        [t7, 0, 3],
    ]

    cube = Cube(x)

    for tr, l, r in trans_cube:
        c1 = cube.transformed(tr)
        draw_show_figs([cube, c1], colors, l, r)

    ccube = CubeCut(y)
    draw_show_figs([ccube], colors, 0, 3)

    plt.show()
