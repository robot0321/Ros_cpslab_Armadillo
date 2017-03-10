// Ros_cpslab_Armadillo.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"
#include <iostream>
#include <Windows.h>
#include <algorithm>
#include <cmath>
#include "CCDPmap.h"
#include <iomanip>


#define pi 3.141592


float Rs = 7; 
float ThetaS = float(57 / 180 * pi);
float sensor_model[2] = { Rs, ThetaS };
int Nside = 4;
float VMAX_const = 4;
float VMIN = -4;
arma::fmat Urng(1, 2); 
arma::fmat Vrng(1, 2); 
const int H = 3; // horizon
const int T = 1;
int ndir = 10;
int nvis = 20;
float s[2];
float phi = 2 * pi / ndir;
float pS[3];
arma::fmat pEst(2, H);
std::vector<float> idxs_x;
std::vector<float> idxs_y;
arma::mat s_dir(1, ndir);
arma::mat phi_a(Nside, 1);
arma::mat cc(Nside, 1);
arma::cube A(Nside, 2, ndir);
arma::mat shiftdir;
std::vector<arma::mat> nextstate_sidxs;
std::vector<arma::mat> nextstate_dir;
CCDPmap* ptrCCDP;


LARGE_INTEGER Frequency;
LARGE_INTEGER BeginTime;
LARGE_INTEGER Endtime;

arma::fmat pseq(2, 5);
Gmap* ptrGmap = new Gmap();
std::vector<arma::mat> Cost_visual;
std::vector<arma::mat> Cost_avoid;
std::vector<arma::cube> Cost_track;
std::vector<arma::cube> Cost_const; // avoid + const


void funf(int a);
double normCDF(double value);
arma::mat unique_rows(arma::mat A);
void setCost_visual(int y, int x);
void setCost_avoid(int y, int x);
void setCost_track(int y, int x, int num);
void setCost_const(int y, int x, int num);
void initCostfunc(int y, int x, int num, float grid);


int main()
{
	
	pseq << 1 << 1 << 1 << 1 << 1 << 4 << 6 << arma::endr
		<< 0 << 1 << 2 << 3 << 4 << 4 << 6 << arma::endr;
	int ttmp = 1;
	Urng(0) = -pi / 2;
	Urng(1) = pi / 2;
	Vrng(0) = VMIN;
	Vrng(1) = VMAX_const;

	s[0] = pseq(0, ttmp) - 1;
	s[1] = pseq(1, ttmp) - 1;
	phi = 2*pi/ndir;
	pS[0] = s[0];
	pS[1] = s[1];
	pS[2] = phi;

	float a = ThetaS / 2;
	for (int i = 0; i < ndir; i++) { s_dir(i) = 2 * PI*i / ndir; }
	
	for (int i = 0; i < Nside; i++) {//setting phi_a
		if (i == 0) { phi_a(0) = (ThetaS / 2 + pi / 2); cc(i) = 0; }
		else if (i == 1) { phi_a(1) = (-ThetaS / 2 - pi / 2); cc(i) = 0; }
		else { phi_a(i) = ((ThetaS*(2 * i + 2) - 3 - Nside) / (2 * (Nside - 2))); cc(i) = Rs*cos(ThetaS / 2 / (Nside - 2)); }
	}
	
	for (int i = 0; i < ndir; i++) {
		A.slice(i).col(0) = cos(phi_a.col(0) + s_dir(i));
		A.slice(i).col(1) = sin(phi_a.col(0) + s_dir(i));
	}

	shiftdir = arma::unique(arma::round(arma::linspace(Urng[0], Urng[1], 20)/2/pi*ndir));
	arma::mat kron_tmp1(2, 20);
	kron_tmp1.row(0) = arma::linspace<arma::rowvec>(0, 1, 20);
	kron_tmp1.row(1) = arma::linspace<arma::rowvec>(1, 0, 20);
	arma::mat kron_tmp2(2, 1);

	for (int i=0; i < ndir; i++) {
		kron_tmp2(0) = sin(s_dir(i));
		kron_tmp2(1) = cos(s_dir(i));
		
		nextstate_sidxs.push_back(unique_rows(arma::trans(arma::round(arma::kron(Vrng*kron_tmp1, kron_tmp2)))));
		arma::mat shiftdir_buffer(shiftdir);
		for (int j = 0; j < shiftdir.size(); j++){
			shiftdir_buffer(j) = (int(shiftdir(j)) + i + ndir) % ndir;
			if (shiftdir_buffer(j) == 0) shiftdir_buffer(j) = ndir;
		}
		nextstate_dir.push_back(shiftdir_buffer);
		//std::cout << (shiftdir + i) << std::endl;
		//std::cout << (arma::mat(shiftdir).fill(ndir)) << std::endl;
		//std::cout << nextstate_dir[i] << std::endl;
	}
	



	QueryPerformanceFrequency(&Frequency);
	system("PAUSE");
	std::cout << pseq(1,0) << std::endl;

	ptrGmap->printGlobalmap();
	//std::cout << std::fixed << std::setprecision(3) << std::setfill('0');

	system("PAUSE");
	for (int timestamps = 0; timestamps < pseq.n_cols-H; timestamps++) {
		QueryPerformanceCounter(&BeginTime);
		
		funf(timestamps);
		
		QueryPerformanceCounter(&Endtime);
		int64_t elapsed = Endtime.QuadPart - BeginTime.QuadPart;
		double duringtime = (double)elapsed / (double)Frequency.QuadPart;
		std::cout << "timestamps: " << timestamps << " takes " << duringtime << " seconds." << std::endl;
		system("PAUSE");
	}
    return 0;
}

void funf(int timestamps) {
	float pT_est[2] = { pseq(0,timestamps), pseq(1,timestamps) };

	for (int i = 0; i < H; i++) {
		pEst.col(i) = pseq.col(timestamps+i+1);
	}	
	std::cout << "pEst " << pEst << std::endl;
	idxs_x = { std::max(std::min(arma::min(pEst.row(0)), pS[0] - Vrng[1] * T), ptrGmap->getXrng().at(0)), std::min(std::max(arma::max(pEst.row(0)), pS[0] + Vrng[1] * T), ptrGmap->getXrng().at(1)) };
	idxs_y = { std::max(std::min(arma::min(pEst.row(1)), pS[1] - Vrng[1] * T), ptrGmap->getYrng().at(0)), std::min(std::max(arma::max(pEst.row(1)), pS[1] + Vrng[1] * T), ptrGmap->getYrng().at(1)) };

	float rlen = idxs_y[1] - idxs_y[0];
	float clen = idxs_x[1] - idxs_x[0];
	std::cout << "idxs_y: " << idxs_y[0] << " / " << idxs_y[1] << std::endl;
	std::cout << "idxs_x: " << idxs_x[0] << " / " << idxs_x[1] << std::endl;

	ptrCCDP = new CCDPmap(ptrGmap, idxs_x, idxs_y, 0.3);
	
	//ptrCCDP->printLocalmap();
	
	initCostfunc(rlen, clen, H, ptrCCDP->getGrid());
	






	delete ptrCCDP;

	system("PAUSE");

}

double normCDF(double value)
{
	return 0.5 * erfc(-value);
}

arma::mat unique_rows(arma::mat A) {
	std::vector<arma::mat> list_of_mat;
	int listchk = 1;
	list_of_mat.push_back(A.row(0));
	for (int j = 0; j < A.n_rows; j++) {
		for (int i = 0; i < list_of_mat.size(); i++) {
			if ((A.row(j))(0) == list_of_mat[i].at(0) && (A.row(j))(1) == list_of_mat[i].at(1)) {
				listchk = 0;
				break;
			}
		}
		if (listchk == 1) {
			list_of_mat.push_back(A.row(j));
		}
		listchk = 1;
	}
	arma::mat Ret(list_of_mat.size(), 2);
	for (int i = 0; i < list_of_mat.size(); i++) {
		Ret.row(i) = list_of_mat[i];
	}

	return Ret;
}


void setCost_visual(int y, int x) {


}
void setCost_avoid(int y, int x) {

}
void setCost_track(int y, int x, int num) {

}
void setCost_const(int y, int x, int num) {

}


void initCostfunc(int y, int x, int num, float grid) {
	arma::mat local = ptrCCDP->getLocalmap();
	for (int i = 0; i < H + 1; i++) {
		Cost_visual.push_back(arma::mat(local).zeros());
		Cost_avoid.push_back(arma::mat(local).zeros());
		Cost_track.push_back(arma::cube(local.n_rows, local.n_cols, ndir));
		Cost_const.push_back(arma::cube(local.n_rows, local.n_cols, ndir));
	}
	arma::mat dist2target(local);
	arma::mat angle2target(local);
	arma::mat minradius = arma::mat(1, nvis).zeros();
	arma::mat cvistmp(local);

	arma::mat tmp(local);

	for (int tt = 0; tt < H; tt++) {
		/****************************** [ Cvis ] *******************************/
		for (int i = 0; i < dist2target.n_rows; i++) {
			for (int j = 0; j < dist2target.n_cols; j++) {
				dist2target(i, j) = pow(pEst(0, tt) - (ptrCCDP->getGrid()*j + idxs_x[0]), 2) + pow(pEst(1, tt) - (ptrCCDP->getGrid()*i + idxs_y[0]), 2);
			}
		}
		std::cout << dist2target << std::endl;

		for (int i = 0; i < dist2target.n_rows; i++) {
			for (int j = 0; j < dist2target.n_cols; j++) {
				angle2target(i, j) = round((std::atan2(pEst(1, tt) - (ptrCCDP->getGrid()*i + idxs_y[0]), pEst(0, tt) - (ptrCCDP->getGrid()*j + idxs_x[0]))+pi)/2/pi*nvis)+1;
				if (angle2target(i, j) == nvis + 1) angle2target(i,j) = 1;
			}
		}
		
		std::cout << angle2target << std::endl;
		int* obsidx = new int[local.n_cols * local.n_rows];
		for (int ii = 0; ii < nvis; ii++) {
			float dist_buff = pow(ptrGmap->getXrng().at(1) - ptrGmap->getXrng().at(0), 2) + pow(ptrGmap->getYrng().at(1) - ptrGmap->getYrng().at(0), 2);
			std::cout << "dist_buff " << ii << " : ";
			int sum_obsidx = 0;
			for (int j = 0; j < local.n_rows*local.n_cols; j++) {
				obsidx[j] = (local(j) == 1 & angle2target(j) == ii);
				tmp(j) = obsidx[j];
				sum_obsidx += obsidx[j];
				if (obsidx[j] == 1) {
					dist_buff = std::fmin(dist2target(j), dist_buff);

					std::cout << dist_buff << " ";
				}
			}
			std::cout << std::endl;
			std::cout << tmp << std::endl;

			if (sum_obsidx > 0) {
				minradius(ii) = dist_buff;
				for (int j = 0; j < local.n_rows*local.n_cols; j++) {
					cvistmp(j) = 1 * ( (dist2target(j) >= minradius(ii)) & (angle2target(j)==ii) );
				}
			}
			std::cout << std::endl;
		}
		delete obsidx;
		Cost_visual[tt+1] = cvistmp;
		//std::cout << "tt: " << tt << " / Cvis : " << std::endl;
		//std::cout << cvistmp << std::endl;


		
		/****************************** [ Ctrack ] *******************************

		for (int ii = 0; ii < ndir; ii++) {
			arma::uvec slice = arma::linspace<arma::uvec>(1, ndir, ndir);
			Cost_track[tt + 1].each_slice(slice) = 


			ccdp.ctrack(tt + 1).val(:, : , ii) = reshape(sum(1 - normcdf(bsxfun(@plus,A(:, : , ii)*[ccdp.xs(:) - pEst(tt).mean(1), ccdp.ys(:) - pEst(tt).mean(2)]',cc)/pEst(tt).sig),1) > threshold_track,[rlen,clen]);% matrix of normal vectors of sensing region
			Cost_track[tt+1]
		}
		

		/****************************** [ Cavoid ] *******************************
		int Rfree = 2; % probability of collision is determined by Rfree
		ccdp.cavoid(:, : , tt + 1) = conv2(double(ccdp.gmap), ones(Rfree * 2 + 1), 'same') >= 1;

		/****************************** [ Cconst ] *******************************
		ccdp.cconst(tt + 1).val = bsxfun(@or,ccdp.ctrack(tt + 1).val, ccdp.cavoid(:, : , tt + 1));
		


		*/
		
	}
}