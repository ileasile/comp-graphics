#pragma once

//#include <armadillo.h>
#include <initializer_list>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include <cmath>
#define PI 3.1415926

class Point;
class Line;
class Triangle;
class ParametricShape;
class ParametricShapeDrawable;
class ParametricShapeTranslator;

class Drawable {
public:
	virtual void draw() = 0;
};

class Point {
	
private:

	static boost::numeric::ublas::matrix<double> inverse(boost::numeric::ublas::matrix<double> a) {
		namespace blas = boost::numeric::ublas;

		blas::matrix<double> Ainv = blas::identity_matrix<double>(a.size1());
		blas::permutation_matrix<size_t> pm(a.size1());
		blas::lu_factorize(a, pm);
		blas::lu_substitute(a, pm, Ainv);

		return Ainv;
	}

	// Solves linear system Ax = b
	static boost::numeric::ublas::vector<double> 
		solve_system(
			boost::numeric::ublas::matrix<double> & a, 
			boost::numeric::ublas::vector<double> & b) {
		namespace blas = boost::numeric::ublas;
		auto Ainv = inverse(a);
		return blas::prod(Ainv, b);
	}
	
public:
	double x, y;
	Point(){}
	Point(double x, double y): x(x), y(y){}

	Point get_symmetric_pt(const Line & l);
	Point get_symmetric_pt_2(const Line & l);
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

class ParametricShape {
public:
	virtual Point pt(double t) const = 0;
	double x(double t) const{
		return pt(t).x;
	}
	double y(double t) const{
		return pt(t).y;
	}
	ParametricShapeDrawable get_drawable(double a, double b, double h);
	ParametricShapeTranslator transform(
		const boost::numeric::ublas::matrix<double> & A,
		const boost::numeric::ublas::vector<double> & b);
	ParametricShapeTranslator shift(double shift_x, double shift_y);
};

class ParametricShapeTranslator: public ParametricShape {
	ParametricShape *shape;
	boost::numeric::ublas::matrix<double> A;
	boost::numeric::ublas::vector<double> b;
public:
	ParametricShapeTranslator(){}

	ParametricShapeTranslator(
		ParametricShape * shape,
		const boost::numeric::ublas::matrix<double> & A,
		const boost::numeric::ublas::vector<double> & b) {
		
		this->shape = shape;
		this->A = A;
		this->b = b;
	}

	virtual Point pt(double t) const{
		auto p = shape->pt(t);
		boost::numeric::ublas::vector<double> x(2);
		x(0) = p.x; x(1) = p.y;
		auto y = boost::numeric::ublas::prod(A, x) + b;
		return Point(y(0), y(1));
	}
};

class ParametricShapeDrawable : public Drawable {
	const ParametricShape * shape;
	double a, b, h;
public:
	ParametricShapeDrawable() {}

	ParametricShapeDrawable(const ParametricShape & shape, double a, double b, double h) {
		this->shape = &shape;
		this->a = a;
		this->b = b;
		this->h = h;
	}

	void set_shape(const ParametricShape * shape) {
		this->shape = shape;
	}

	virtual void draw() {
		Point fr, to;
		fr = shape->pt(a);
		for (double t = a+h; t <= b; t += h) {
			to = shape->pt(t);
			draw_line(fr, to);
			fr = to;
		}
	}
};

class CEllipse : public ParametricShape {
	double a, b;
public:
	CEllipse() {}
	CEllipse(double a, double b): a(a), b(b){}
	virtual Point pt(double t) const{
		return Point(a * cos(t), b * sin(t));
	}
	ParametricShapeDrawable get_drawable(int n_pt = 100) {
		double h = 2 * PI / n_pt;
		return ParametricShape::get_drawable(0, 2 * PI + h, h);
	}
};