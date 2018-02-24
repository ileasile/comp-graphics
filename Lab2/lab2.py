import numpy as np
from math import sin, cos, atan2, sqrt
from qhull_convex import qconvex
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import mpl_backend_workaround
import matplotlib.pyplot as plt


def rotate_mtx(ax=0., ay=0., az=0.):
    x = np.array([
        [1, 0,          0,          0],
        [0, cos(ax),    sin(ax),    0],
        [0, -sin(ax),   cos(ax),    0],
        [0, 0,          0,          1],
    ])
    y = np.array([
        [cos(ay),   0, -sin(ay),    0],
        [0,         1, 0,           0],
        [sin(ay),   0, cos(ay),     0],
        [0,         0, 0,           1],
    ])
    z = np.array([
        [cos(az),   sin(az),    0, 0],
        [-sin(az),  cos(az),    0, 0],
        [0,         0,          1, 0],
        [0,         0,          0, 1],
    ])
    return x @ y @ z


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
t4 = rotate_mtx(ax=np.pi/2)
t5 = perm_mtx([1, 2, 0, 3])
t6 = trans_mtx('z')
t7 = shift_mtx([2, 1, 0])

_t8_shift = shift_mtx([0, -1.5, -1.5])
t8 = _t8_shift @ rotate_mtx(ax=np.pi/6) @ np.linalg.inv(_t8_shift)

t9 = _t8_shift @ rotate_mtx(ax=np.pi/6, ay=np.pi*3/2) @ np.linalg.inv(_t8_shift)


class Shape:
    def __init__(self, surfaces):
        self.s = surfaces

    def transformed(self, transform):
        new_surfaces = []
        for t in self.s:
            new_surfaces.append(t @ transform)
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


class ConvexShape(Shape):
    def __init__(self, x):
        y = x[:, :]
        for i in range(3):
            y[:, i] = x[:, i] / x[:, 3]

        ch = qconvex("i Qi", y[:, :3])
        coordinates = []
        for s in ch[1:]:
            facet = [int(i) for i in s.split(' ')]
            coordinates.append(facet)

        s = [x[c, :] for c in coordinates]
        super().__init__(s)


class Plane(Shape):
    def __init__(self, pts: np.ndarray):
        self.x = pts[:, :3]
        self.x0 = pts[0, :3]
        x1 = pts[1, :3]
        x2 = pts[2, :3]
        u = x1 - self.x0
        v = x2 - self.x0
        self.n = np.cross(u, v)

        s = [pts]
        super().__init__(s)

    def origin_shift(self):
        return shift_mtx((-self.x0).tolist())

    def norm_rotate_oz(self):
        phi = atan2(sqrt(self.n[0]**2 + self.n[1]**2), self.n[2])
        psi = atan2(self.n[1], self.n[0])
        r1 = rotate_mtx(az=-psi)
        r2 = rotate_mtx(ay=-phi)
        return r1 @ r2

    def half_reflection(self):
        return self.origin_shift() @ self.norm_rotate_oz()


class Line:
    def __init__(self, pts):
        self.x = pts[:, :3]
        self.x0 = pts[0, :3]
        x1 = pts[1, :3]
        self.u = x1 - self.x0

    def origin_shift(self):
        return shift_mtx((-self.x0).tolist())

    def dir_rotate_to_oz(self):
        phi = atan2(sqrt(self.u[0]**2 + self.u[1]**2), self.u[2])
        psi = atan2(self.u[1], self.u[0])
        r1 = rotate_mtx(az=-psi)
        r2 = rotate_mtx(ay=-phi)
        return r1 @ r2

    def rotation_mtx(self, alpha):
        d = self.origin_shift() @ self.dir_rotate_to_oz()
        d_inv = np.linalg.inv(d)
        return d @ rotate_mtx(az=alpha) @ d_inv


def draw_show_figs(name: str, figs, cols, lim_l=-4, lim_r=4):
    fig = plt.figure(name)
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+100+100")
    ax = Axes3D(fig)
    ax.set_aspect('equal')
    ax.set_xlim(lim_l, lim_r)
    ax.set_ylim(lim_l, lim_r)
    ax.set_zlim(lim_l, lim_r)

    for i, f in enumerate(figs):
        f.draw(ax, cols[i])

    plt.show()


def main():
    print('Using MPL backend: {}'.format(mpl_backend_workaround.MPL_BACKEND_USED))

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

    trans_cube = [
        [t1, 0, 3],
        [t2, 0, 6],
        [t3, -1, 5],
        [t4, -1, 3],
        [t5, 0, 3],
        [t6, 0, 3],
        [t7, 0, 5],
    ]
    # Uncomment to skip these simple transforms
    # trans_cube = []

    cube = ConvexShape(x)

    for num, (tr, l, r) in enumerate(trans_cube):
        c1 = cube.transformed(tr)
        draw_show_figs("Transform #{}".format(num + 1), [cube, c1], colors, l, r)

    # Transformations 8 - 11 on cut cube
    cut_cube = ConvexShape(y)
    trans_cube_cut = [
        [t8, 0, 5, "Transform #8"],
        [t9, 0, 5, "Transform #9"],
    ]

    for tr, l, r, name in trans_cube_cut:
        c1 = cut_cube.transformed(tr)
        draw_show_figs(name, [cut_cube, c1], colors, l, r)

    # 10-th transform
    line_ax = Line(y[[3, 9], :])
    line_rotation = line_ax.rotation_mtx(-np.pi)
    cut_cube_rotated = cut_cube.transformed(line_rotation)
    draw_show_figs("Rotate around some axis", [cut_cube, cut_cube_rotated], colors, 0, 3)

    # Last transform
    pl = Plane(y[[2, 3, 9], :])

    half_ref = pl.half_reflection()
    half_ref_inv = np.linalg.inv(half_ref)
    simple_ref = trans_mtx('z')

    reflection_mtx = np.linalg.multi_dot([half_ref, simple_ref, half_ref_inv])
    cut_cube_reflected = cut_cube.transformed(reflection_mtx)

    pl_tr = pl.transformed(reflection_mtx)

    draw_show_figs("Reflection", [cut_cube, pl_tr, cut_cube_reflected], colors, 0, 4)


if __name__ == "__main__":
    main()
