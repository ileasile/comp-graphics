#include "Shapes.h"

Point Point::get_symmetric_pt(const Line & l) {
	namespace blas = boost::numeric::ublas;
	blas::matrix<double> a(2, 2);
	blas::vector<double> b(2);
	double xp = l.p[1].x - l.p[0].x;
	double yp = l.p[1].y - l.p[0].y;
	a(0, 0) = xp;
	a(0, 1) = yp;
	a(1, 0) = yp;
	a(1, 1) = -xp;
	b(0) = x * xp + y * yp;
	b(1) = xp * y - yp * x + 2 * (l.p[0].x * yp - l.p[0].y * xp);
	auto x = solve_system(a, b);
	return Point(x(0), x(1));
}

Point Point::get_symmetric_pt_2(const Line & l) {
	namespace blas = boost::numeric::ublas;
	blas::matrix<double> a = blas::identity_matrix<double>(3);
	blas::matrix<double> b = blas::identity_matrix<double>(3);
	blas::matrix<double> t = blas::identity_matrix<double>(3);
	blas::vector<double> x_(3);
	x_(0) = x; x_(1) = y; x_(2) = 1;
	
	double xp = l.p[1].x - l.p[0].x;
	double yp = l.p[1].y - l.p[0].y;
	
	double phi = atan2(yp, xp);
	a(0, 0) = a(1, 1) = cos(-phi);
	a(0, 1) = -(a(1, 0) = sin(-phi));
	t(1, 1) = -1;
	b(0, 2) = -l.p[0].x;
	b(1, 2) = -l.p[0].y;

	blas::matrix<double> c = blas::prod(a, b);
	blas::matrix<double> cinv = inverse(c);
	blas::matrix<double> ainv = inverse(a);
	
	blas::vector<double> y = blas::prod(c, x_);
	blas::vector<double> y1 = blas::prod(t, y);
	blas::vector<double> y2 = blas::prod(cinv, y1);

	return Point(y2(0), y2(1));
}

Triangle Triangle::get_symmetric(const Line & l) {
	Point p1[3];
	for (int i = 0; i < 3; ++i) {
		p1[i] = p[i].get_symmetric_pt_2(l);
	}
	return Triangle(p1);
}

void draw_line(const Point & p1, const Point & p2) {
	cinder::gl::drawLine(glm::vec2(p1.x, p1.y), glm::vec2(p2.x, p2.y));
}

ParametricShapeDrawable ParametricShape::get_drawable(double a, double b, double h) {
	return ParametricShapeDrawable(*this, a, b, h);
}

ParametricShapeTranslator ParametricShape::transform(
	const boost::numeric::ublas::matrix<double>& A, 
	const boost::numeric::ublas::vector<double>& b)
{
	return ParametricShapeTranslator(this, A, b);
}

ParametricShapeTranslator ParametricShape::shift(double shift_x, double shift_y) {
	boost::numeric::ublas::identity_matrix<double> A(2);
	boost::numeric::ublas::vector<double> b(2);
	b(0) = shift_x; b(1) = shift_y;
	return transform(A, b);
}
