#include "stdafx.h"
#include "CCDPmap.h"
#include <iostream>

CCDPmap::CCDPmap(Gmap* ptr_gmap, std::vector<float> x_rng, std::vector<float> y_rng, int rlen, int clen, int H, float epsilon=0.3) {
	grid = ptr_gmap->getGrid();
	ndir = 10;
	nvis = 20;
	H_step = 10;
	threshold_track = epsilon;
	
	
	CCDPmap::setLocalmap(ptr_gmap, x_rng, y_rng);
	CCDPmap::initCostfunc(rlen, clen, H);
}

CCDPmap::~CCDPmap() {
	delete local_map;
	delete Cost_visual;
	delete Cost_avoid;
	delete Cost_track;
	delete Cost_const;
}

void CCDPmap::grid_resize(int factor) {
	int new_x_num = local_num_x/factor;
	int new_y_num = local_num_y/factor;
	arma::fmat* new_map_tmp = new arma::fmat(new_y_num, new_x_num);

	for (int i = 0; i < new_y_num; i++) {
		for (int j = 0; j < new_x_num; j++) {
			new_map_tmp->at(i, j) = local_map->at(2 * i, 2 * j); //Need to edit the resizing way
		}
	}
	//delete memory of local_map pointer
	delete local_map;

	local_map = new_map_tmp;
	grid = grid*factor;
	local_num_x = new_x_num;
	local_num_y = new_y_num;

}

void CCDPmap::setLocalmap(Gmap* ptr_gmap, std::vector<float> x_rng, std::vector<float> y_rng) {
	local_x_rng = x_rng;
	local_y_rng = y_rng;
	local_x_len = local_x_rng[1] - local_x_rng[0];
	local_y_len = local_y_rng[1] - local_y_rng[0];
	local_num_x = local_x_len / grid;
	local_num_y = local_y_len / grid;
	
	arma::fmat* global_map = ptr_gmap->getGlobalmap();
	std::vector<float> global_Xrng = ptr_gmap->getXrng();
	std::vector<float> global_Yrng = ptr_gmap->getYrng();

	local_map = new arma::fmat(local_num_y, local_num_x);
	for (int i = 0; i < local_num_y; i++) {
		for (int j = 0; j < local_num_x; j++) {
			local_map->at(i,j) = global_map->at(i + int((local_y_rng[0] - global_Yrng[0]) / grid), j + int((local_x_rng[0] - global_Xrng[0]) / grid));
		}
	}
}

arma::fmat* CCDPmap::getLocalmap() const {return local_map;}

void CCDPmap::printLocalmap() const {
	for (int i = 0; i < local_num_y; i++) {
		for (int j = 0; j < local_num_x; j++) {
			if (i == -int(local_y_rng[0] / grid)) {
				std::cout << "^";
			}
			else if (j == -int(local_x_rng[0] / grid)) {
				std::cout << "<";
			}
			else {
				std::cout << local_map->at(local_num_y - 1 - i, j);
			}
		}
		std::cout << std::endl;
	}
}

arma::fmat* CCDPmap::getCost_visual() const {return Cost_visual;}
arma::fmat* CCDPmap::getCost_avoid() const {return Cost_avoid;}
arma::cube* CCDPmap::getCost_track() const {return Cost_track; }
arma::cube* CCDPmap::getCost_const() const {return Cost_const;}

void CCDPmap::setCost_visual(int y, int x) {
	Cost_visual = new arma::fmat(y, x);
}
void CCDPmap::setCost_avoid(int y, int x) {
	Cost_avoid = new arma::fmat(y, x);
}
void CCDPmap::setCost_track(int y, int x, int num) {
	Cost_track = new arma::cube(y, x, num);
}
void CCDPmap::setCost_const(int y, int x, int num) {
	Cost_const= new arma::cube(y, x, num);
}


void CCDPmap::initCostfunc(int y, int x, int num) {
	setCost_visual(y, x);
	setCost_avoid(y, x);
	setCost_track(y, x, num);
	setCost_const(y, x, num);
}

/*

int main() {
	std::vector<float> x = { -10,10 };
	std::vector<float> y = { -10,10 };
	system("PAUSE");
	Gmap* a = new Gmap();
	a->printGlobalmap();
	system("PAUSE");
	CCDPmap* b = new CCDPmap(a, x, y, 0.3);
	b->printLocalmap();
	system("PAUSE");
	b->grid_resize(3);
	b->printLocalmap();
	system("PAUSE");
	

	delete b;
	delete a;


	return 0;
}

*/