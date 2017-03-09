#pragma once
#include <vector>
#include "Gmap.h"

#ifndef PI
	#define PI 3.141593
#endif // !


class CCDPmap { //get Local CCDP map
private:
	std::vector<float> local_x_rng;
	std::vector<float> local_y_rng;
	float local_x_len;
	float local_y_len;
	int local_num_x;
	int local_num_y;
	float grid;
	int ndir;
	int nvis;
	int H_step;
	float threshold_track;
	
	


	arma::fmat* local_map;
	arma::fmat* Cost_visual;
	arma::fmat* Cost_avoid;
	arma::cube* Cost_track;
	arma::cube* Cost_const; // avoid + const

public:
	CCDPmap(Gmap* ptr_gmap, std::vector<float> x_rng, std::vector<float> y_rng, int rlen, int clen, int H, float epsilon);
	~CCDPmap();


	void grid_resize(int factor);

	void setLocalmap(Gmap* ptr_gmap, std::vector<float> x_rng, std::vector<float> y_rng);
	arma::fmat* getLocalmap() const;
	void printLocalmap() const;

	void setCost_visual(int y, int x);
	void setCost_avoid(int y, int x);
	void setCost_track(int y, int x, int num);
	void setCost_const(int y, int x, int num);

	arma::fmat* getCost_visual() const;
	arma::fmat* getCost_avoid() const;
	arma::cube* getCost_track() const;
	arma::cube* getCost_const() const;

	void CCDPmap::initCostfunc(int y, int x, int num);

	//ccdp.xs/ys -> (num-Áß¾Ó)/grid·Î ´ëÃ³
	//ccdp.glen -> grid
	//ccdp.gmap -> local_map


};