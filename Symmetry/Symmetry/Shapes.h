#pragma once

//#include <armadillo.h>
#include <initializer_list>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
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
	static Point solve_system(boost::numeric::ublas::matrix<double> & a, boost::numeric::ublas::vector<double> & b) {
		boost::numeric::ublas::matrix<double> Ainv = boost::numeric::ublas::identity_matrix<double>(a.size1());
		boost::numeric::ublas:: permutation_matrix<size_t> pm(a.size1());
		boost::numeric::ublas::lu_factorize(a, pm);
		boost::numeric::ublas::lu_substitute(a, pm, Ainv);

		auto x = boost::numeric::ublas::prod(Ainv, b);
		return Point(x(0), x(1));
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
	Line(const std::initializer_list<Point> & v) {
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

	Triangle(const std::initializer_list<Point> & v) {
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