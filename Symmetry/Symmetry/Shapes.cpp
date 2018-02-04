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

Triangle Triangle::get_symmetric(const Line & l) {
	Point p1[3];
	for (int i = 0; i < 3; ++i) {
		p1[i] = p[i].get_symmetric_pt(l);
	}
	return Triangle(p1);
}

void draw_line(const Point & p1, const Point & p2) {
	cinder::gl::drawLine(glm::vec2(p1.x, p1.y), glm::vec2(p2.x, p2.y));
}
