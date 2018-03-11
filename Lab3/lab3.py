import numpy as np
from math import sin, cos, sqrt, acosh, cosh, sinh
from mpl_backend_workaround import MPL_BACKEND_USED
import matplotlib.lines as m_lines
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Iterable, Callable, Tuple, Any, List

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

    c = Circle(0, 0, 1, 100)
    cb = Circle(0.5, 0.3, 1.5, 20)
    draw(c, cb, title='Circles')

    el = Ellipse(0, 0, 4, 1, 20)
    tr = rotate_transform(np.pi/6) @ shift_transform(2, 2)
    el_t = el.transformed(tr)
    draw(el, el_t, title='Ellipses')

    parabola = Parabola(.5, 4, 10)
    draw(parabola, title='Parabola')

    h = Hyperbola(2, 1, 5., 20)
    draw(h, title='Hyperbola')


if __name__ == "__main__":
    main()
