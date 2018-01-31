#pragma once

//#include <armadillo.h>
#include <initializer_list>
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

class Point;
class Line;
class Triangle;


class Drawable {
public:
	virtual void draw() = 0;
};

class Point {
private:
	// Solves 2-range linear system
	static Point solve_system(double * a, double * b) {
		double d = a[0] * a[3] - a[1] * a[2];
		double d1 = b[0] * a[3] - b[1] * a[1];
		double d2 = a[0] * b[1] - b[0] * a[2];
		return Point(d1 / d, d2 / d);
	}
public:
	double x, y;
	Point(){}
	Point(double x, double y): x(x), y(y){}

	Point get_symmetric_pt(const Line & l);
};

void draw_line(const Point & p1, const Point & p2);

class Line : public Drawable {
public:
	Point p[2];
	Line() {}
	Line(Point * p1) {
		for (int i = 0; i < 2; ++i) {
			p[i] = p1[i];
		}
	}
	Line(std::initializer_list<Point> v) {
		Point * pb = p;
		for (auto & pt : v) {
			*(pb++) = pt;
		}
	}
	virtual void draw() {
		draw_line(p[0], p[1]);
	}
};

class Triangle : public Drawable {
public:
	Point p[3];
	Triangle(){}

	Triangle(Point * p1) {
		for (int i = 0; i < 3; ++i) {
			p[i] = p1[i];
		}
	}

	Triangle(std::initializer_list<Point> v) {
		Point * pb = p;
		for (auto & pt : v) {
			*(pb ++) = pt;
		}
	}
	
	Triangle get_symmetric(const Line & l);

	virtual void draw() {
		for (int i = 0; i < 3; ++i) {
			int j = (i + 1) % 3;
			draw_line(p[i], p[j]);
		}
	}
};