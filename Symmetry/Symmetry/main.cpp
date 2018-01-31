#include <iostream>

#include "GraphicException.h"
#include "Shapes.h"

class SymmetryApp : public ci::app::App {
	Triangle tr, tr2;
	Line l;
	ci::Color red, green, blue, white;
public:
	SymmetryApp();
	void draw() override;
	static void draw(Drawable * d, ci::Color & c, float width) {
		ci::gl::color(c);
		ci::gl::lineWidth(width);
		d->draw();
	}
};

SymmetryApp::SymmetryApp() {
	tr = { { 2, 4 },{ 4, 6 },{ 2, 6 } };
	double k = .5, b = 2.;

	int x_max = 800, y_max = 600;
	l = { { 0, b },{ double(x_max), x_max * k + b } };
	tr2 = tr.get_symmetric(l);
	red = ci::Color(1.f, 0.f, 0.f);
	green = ci::Color(0.f, 1.f, 0.f);
	blue = ci::Color(0.f, 0.f, 1.f);
	white = ci::Color(1.f, 1.f, 1.f);
	this->setWindowSize(x_max, y_max);
}

void SymmetryApp::draw()
{
	ci::gl::clear(white);
	ci::gl::pushModelMatrix();
	ci::gl::scale(90., 90.);
	draw(&l, red, 3);
	draw(&tr, green, 3);
	draw(&tr2, blue, 3);
	ci::gl::popModelMatrix();
}

CINDER_APP(SymmetryApp, ci::app::RendererGl)
