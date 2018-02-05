#include <iostream>

#include "GraphicException.h"
#include "Shapes.h"
#include <INIReader.h>

class SymmetryApp : public ci::app::App {
	Triangle tr, tr2;
	Line l;
	ParametricShapeDrawable d_el;
	CEllipse * el;
	ParametricShapeTranslator tr_el;
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
	INIReader conf("config.ini");
	double x[3];
	double y[3];
	for (int i = 0; i < 3; ++i) {
		auto s = std::to_string(i + 1);
		x[i] = conf.GetReal("Task", "x" + s, 0);
		y[i] = conf.GetReal("Task", "y" + s, 0);
	}

	tr = { { x[0], y[0] },{ x[1], y[1] },{ x[2], y[2] } };
	double	k = conf.GetReal("Task", "k", .5), 
			b = conf.GetReal("Task", "b", 2.);

	int x_max = 800, y_max = 600;
	l = { { 0, b },{ double(x_max), x_max * k + b } };
	tr2 = tr.get_symmetric(l);

	el = new CEllipse(3., 4.);
	tr_el = el->shift(3, 4);
	d_el = el->get_drawable();
	d_el.set_shape(&tr_el);

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
	//draw(&d_el, red, 4);
	ci::gl::popModelMatrix();
}

CINDER_APP(SymmetryApp, ci::app::RendererGl)
