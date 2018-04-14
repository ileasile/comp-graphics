import numpy as np
from math import sin, cos, sqrt, acosh, cosh, sinh
from mpl_backend_workaround import MPL_BACKEND_USED
import matplotlib.lines as m_lines
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Iterable, Callable, Tuple, Any, List
from config import CONFIG
from scipy.special import binom
from numpy.polynomial.polynomial import polyval

IterFunc = Callable[[float, float], float]


class Drawable:
    def transformed(self, transform: np.ndarray) -> 'Drawable':
        raise NotImplementedError()

    def b_box(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        raise NotImplementedError()

    def draw(self, ax: Axes, col) -> None:
        raise NotImplementedError()


class Shape(Drawable):
    def __init__(self, points: np.ndarray, do_close: bool):
        self.p = points
        self.close = do_close

    def transformed(self, transform: np.ndarray) -> 'Shape':
        new_points = self.p @ transform
        return Shape(new_points, self.close)

    def b_box(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        x = self.p[:, 0] / self.p[:, 2]
        y = self.p[:, 1] / self.p[:, 2]

        return (np.min(x), np.min(y)), (np.max(x), np.max(y))

    def draw(self, ax: Axes, col) -> None:
        pt = np.transpose(self.p)
        if self.close:
            pt = np.hstack([pt, pt[:, :1]])

        x, y = pt[:2, :]
        x /= pt[2, :]
        y /= pt[2, :]

        poly_line = m_lines.Line2D(x, y, color=col)
        ax.add_line(poly_line)


class IterativeLine(Shape):
    def __init__(self, x0: float, y0: float, 
                 xf: IterFunc,
                 yf: IterFunc, n: int, close: bool):
        p = np.ones((n, 3))
        p[0, :2] = [x0, y0]
        for i in range(1, n):
            x = p[i - 1, 0]
            y = p[i - 1, 1]
            p[i, :2] = [xf(x, y), yf(x, y)]

        super().__init__(p, close)


class IterativeShiftedLine(IterativeLine):
    def __init__(self, x0: float, y0: float, xf: IterFunc, yf: IterFunc, n: int, close: bool,
                 x_shift: float, y_shift: float):
        super().__init__(x0, y0, xf, yf, n, close)
        self.p[:, 0] = self.p[:, 0] + x_shift
        self.p[:, 1] = self.p[:, 1] + y_shift


class Ellipse(IterativeShiftedLine):
    def __init__(self, x0: float, y0: float, a: float, b: float, n: int = 10):
        phi = 2 * np.pi / n
        cf = cos(phi)
        sf = sin(phi)

        a_b = float(a) / b
        sf_a_b = sf * a_b
        sf_b_a = sf / a_b

        super().__init__(a, 0,
                         lambda x, y: x * cf - y * sf_a_b,
                         lambda x, y: x * sf_b_a + y * cf,
                         n, True,
                         x0, y0)


class Circle(Ellipse):
    def __init__(self, x0: float, y0: float, r: float, n: int=10):
        super().__init__(x0, y0, r, r, n)


class ParabolaBranch(IterativeLine):
    def __init__(self, a: float, x_max: float, n: int):
        t_max = sqrt(float(x_max) / a)
        dt = t_max / (n-1)
        super().__init__(0, 0,
                         lambda x, y: x + y * dt + a * dt * dt,
                         lambda x, y: y + 2 * a * dt,
                         n, False)


class HyperbolaBranch(IterativeLine):
    def __init__(self, a: float, b: float, x_max: float, n: int):
        t_max = acosh(x_max / a)
        dt = t_max / (n - 1)

        cf = cosh(dt)
        sf = sinh(dt)

        a_b = float(a) / b
        sf_a_b = sf * a_b
        sf_b_a = sf / a_b

        super().__init__(1, 0,
                         lambda x, y: x * cf + y * sf_a_b,
                         lambda x, y: x * sf_b_a + y * cf,
                         n, False)


class MultiShape(Drawable):
    def __init__(self, shapes: Iterable[Drawable]):
        self.s = shapes

    def transformed(self, transform: np.ndarray) -> 'MultiShape':
        s = [c.transformed(transform) for c in self.s]
        return MultiShape(s)

    @staticmethod
    def bounding_box(s: Iterable[Drawable]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        b = [f.b_box() for f in s]
        x_min = min([t[0][0] for t in b])
        y_min = min([t[0][1] for t in b])
        x_max = max([t[1][0] for t in b])
        y_max = max([t[1][1] for t in b])
        return (x_min, y_min), (x_max, y_max)

    def b_box(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return MultiShape.bounding_box(self.s)

    def draw(self, ax, col) -> None:
        for f in self.s:
            f.draw(ax, col)


class Hyperbola(MultiShape):
    def __init__(self, a: float, b: float, x_max: float, n: int):
        b = HyperbolaBranch(a, b, x_max, n)
        s = [b]

        t = np.identity(3)
        t[0, 0] = -1
        s.append(b.transformed(t))

        t[1, 1] = -1
        s.append(b.transformed(t))

        t[0, 0] = 1
        s.append(b.transformed(t))

        super().__init__(s)


class Parabola(MultiShape):
    def __init__(self, a: float, x_max: float, n: int):
        up_branch = ParabolaBranch(a, x_max, n)
        t = np.identity(3)
        t[1, 1] = -1
        lo_branch = up_branch.transformed(t)
        super().__init__([up_branch, lo_branch])


class BezierCurve(Shape):
    def __init__(self, points: List[Tuple[float, float]], n_steps: int = 100):
        p = np.ones((n_steps, 3))

        b_poly = [np.poly1d([])] * 2
        t_poly = np.poly1d([1., 0.])
        t_1_poly = np.poly1d([-1., 1.])
        n = len(points) - 1
        for i, pt in enumerate(points):
            pt_a = binom(n, i) * np.array(pt)
            b = (t_poly ** i) * (t_1_poly ** (n - i))
            for j in range(2):
                b_poly[j] += np.poly1d([pt_a[j]]) * b

        coefficients = np.zeros((n + 1, 2), dtype=np.float64)
        for i in range(2):
            coefficients[: b_poly[i].order + 1, i] = b_poly[i].coef[::-1]
        p[:, :2] = polyval(np.linspace(0, 1, n_steps), coefficients, tensor=True).transpose()

        super().__init__(p, False)


class SplineCondition:
    def a_apply(self, a: np.ndarray, t: np.ndarray):
        raise NotImplementedError()

    def b_apply(self, b: np.ndarray, t: np.ndarray, p: np.ndarray):
        raise NotImplementedError()

    def get_point_diff(self, a: np.ndarray, b: np.ndarray, t: np.ndarray, p: np.ndarray):
        self.a_apply(a, t)
        self.b_apply(b, t, p)
        return np.linalg.inv(a) @ b


class SplineFixedCondition(SplineCondition):
    def __init__(self, p1d, pnd):
        self.p1d = p1d
        self.pnd = pnd

    def a_apply(self, a: np.ndarray, t: np.ndarray):
        a[0, 0] = 1
        a[-1, -1] = 1

    def b_apply(self, b: np.ndarray, t: np.ndarray, p: np.ndarray):
        b[0, :] = self.p1d
        b[-1, :] = self.pnd


class SplineWeakCondition(SplineCondition):
    def a_apply(self, a: np.ndarray, t: np.ndarray):
        a[0, 0] = 1
        a[0, 1] = .5
        a[-1, -2] = 2
        a[-1, -1] = 4

    def b_apply(self, b: np.ndarray, t: np.ndarray, p: np.ndarray):
        b[0, :] = 3.*(p[1, :] - p[0, :]) / (2 * t[0])
        b[-1, :] = 6.*(p[-1, :] - p[-2, :])/t[-1]


class SplineCyclicCondition(SplineCondition):
    def a_apply(self, a: np.ndarray, t: np.ndarray):
        a[0, 0] = 2 * (1 + t[-1] / t[0])
        a[0, 1] = t[-1] / t[0]
        a[0, -2] = 1

    def b_apply(self, b: np.ndarray, t: np.ndarray, p: np.ndarray):
        b[0, :] = 3.*((t[-1] / t[0] ** 2) * (p[1, :] - p[0, :]) - (p[-2, :] - p[-1, :])/t[-1])

    def get_point_diff(self, a: np.ndarray, b: np.ndarray, t: np.ndarray, p: np.ndarray):
        self.a_apply(a, t)
        self.b_apply(b, t, p)
        pd = np.zeros((a.shape[0], 2), dtype=a.dtype)
        pd[:-1, :] = np.linalg.inv(a[:-1, :-1]) @ b[:-1, :]
        pd[-1, :] = pd[0, :]
        return pd


class SplineAcyclicCondition(SplineCyclicCondition):
    def a_apply(self, a: np.ndarray, t: np.ndarray):
        a[0, 0] = 2 * (1 + t[-1] / t[0])
        a[0, 1] = t[-1] / t[0]
        a[0, -2] = -1

    def b_apply(self, b: np.ndarray, t: np.ndarray, p: np.ndarray):
        b[0, :] = 3. * ((t[-1] / t[0] ** 2) * (p[1, :] - p[0, :]) + (p[-2, :] - p[-1, :]) / t[-1])

    def get_point_diff(self, a: np.ndarray, b: np.ndarray, t: np.ndarray, p: np.ndarray):
        pd = super().get_point_diff(a, b, t, p)
        pd[-1, :] = -pd[0, :]
        return pd


class Spline(Shape):
    def __init__(self, points: List[Tuple[float, float]], edge_condition: SplineCondition, n_steps: int = 100):
        n = len(points) - 1
        dp = np.zeros((n+1, 2), dtype=np.float)
        for i, tp in enumerate(points):
            dp[i, :] = tp

        t = np.sqrt((dp[1:, 0] - dp[:-1, 0]) ** 2 + (dp[1:, 1] - dp[:-1, 1]) ** 2)

        a = np.zeros((n + 1, n + 1), dtype=np.float)
        for i in range(1, n):
            a[i, i - 1] = t[i]
            a[i, i + 1] = t[i - 1]
            a[i, i] = 2 * (t[i] + t[i - 1])

        b = np.zeros((n + 1, 2), dtype=np.float)
        for i in range(1, n):
            b[i, :] = 3.0/(t[i-1] * t[i]) *\
                      (t[i - 1] ** 2 * (dp[i + 1, :] - dp[i, :]) +
                       t[i] ** 2 * (dp[i, :] - dp[i - 1, :]))

        pd = edge_condition.get_point_diff(a, b, t, dp)

        p_list = []
        n_per_iter = n_steps // n + 1
        for i in range(n):
            taus = np.linspace(0, 1, n_per_iter).reshape((n_per_iter, 1))
            t2 = taus ** 2
            t3 = t2 * taus
            f_vec = np.hstack([2 * t3 - 3 * t2 + 1,
                               -2 * t3 + 3 * t2,
                               taus * (taus - 1)**2 * t[i],
                               taus * (t2 - taus) * t[i]])
            p_vec = np.vstack([dp[i, :], dp[i+1, :], pd[i, :], pd[i+1, :]])
            seg_points = f_vec @ p_vec
            seg_points = np.hstack([seg_points, np.ones((n_per_iter, 1))])
            p_list.append(seg_points)

        p = np.vstack(p_list)

        super().__init__(p, False)


def draw_figs(figs: Iterable[Drawable], colors: List[Any], title: str):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title(title)
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+100+100")

    ax.set_aspect('equal')

    (x_min, y_min), (x_max, y_max) = MultiShape.bounding_box(figs)

    percent_left = 0.1
    x_inc = (x_max - x_min) * percent_left
    y_inc = (y_max - y_min) * percent_left

    ax.set_xlim(x_min - x_inc, x_max + x_inc)
    ax.set_ylim(y_min - y_inc, y_max + y_inc)

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    ax.grid(True, which='both')

    for i, fig in enumerate(figs):
        fig.draw(ax, colors[i % len(colors)])

    plt.show()


def rotate_transform(phi):
    return np.array([
        [cos(phi), sin(phi), 0],
        [-sin(phi), cos(phi), 0],
        [0, 0, 1]
    ])


def shift_transform(x, y):
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [x, y, 1]
    ])


def main():
    print('Using MPL backend: {}'.format(MPL_BACKEND_USED))

    colors = ['red', 'blue', 'green']

    def draw(*figs, **kwargs):
        title = 'Figure' if 'title' not in kwargs else kwargs['title']
        draw_figs(figs, colors, title)

    if CONFIG['draw_simple_lines']:
        c = Circle(0, 0, 1, 100)
        cb = Circle(0.5, 0.3, 1.5, 20)
        draw(c, cb, title='Circles')

        el = Ellipse(0, 0, 4, 1, 20)
        tr = rotate_transform(np.pi/6) @ shift_transform(2, 2)
        el_t = el.transformed(tr)
        draw(el, el_t, title='Ellipses')

        parabola = Parabola(.5, 4, 10)
        tr = rotate_transform(np.pi / 4) @ shift_transform(1, 1)
        parabola_t = parabola.transformed(tr)

        draw(parabola_t, title='Parabola')

        h = Hyperbola(2, 1, 5., 20)
        draw(h, title='Hyperbola')

    if CONFIG['draw_bezier']:
        # bezier_linear = BezierCurve([(0, 0), (1, 3)])
        # draw(bezier_linear, title='Linear bezier curve')

        # bezier_quadratic = BezierCurve([(0, 0), (2, -5), (4, 0)])
        # draw(bezier_quadratic, title='Quadratic bezier curve')

        bezier_cubic = BezierCurve([(-3, 0), (0, 2), (2, 6), (4, -3)])
        draw(bezier_cubic, title='Cubic bezier curve')

    if CONFIG['draw_splines']:
        conditions = [
            (SplineFixedCondition((1, 1), (1, 1)), "fixed condition"),
            (SplineWeakCondition(), "weak condition"),
            (SplineCyclicCondition(), "cyclic condition"),
            (SplineAcyclicCondition(), "acyclic condition"),
        ]

        for c, half_title in conditions:
            cubic_spline = Spline([(-3, 0), (0, 2), (2, 6), (4, -3)], c, 100)
            draw(cubic_spline, title='Cubic spline: {}'.format(half_title))


if __name__ == "__main__":
    main()
