// Ros_cpslab_Armadillo.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"
#include <iostream>
#include <Windows.h>
#include <algorithm>
#include <cmath>
#include "CCDPmap.h"

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
arma::fmat s_dir(1, ndir);
arma::fmat phi_a(Nside, 1);
arma::fmat cc(Nside, 1);
arma::fcube A(Nside, 2, ndir);
std::vector<float> shiftdir;
std::vector<arma::mat> nextstate_sidxs;
std::vector<arma::mat*> nextstate_dir;



LARGE_INTEGER Frequency;
LARGE_INTEGER BeginTime;
LARGE_INTEGER Endtime;

arma::fmat pseq(2, 5);
Gmap* ptrGmap = new Gmap();

void funf(int a);
double normCDF(double value)
{
	return 0.5 * erfc(-value);
}


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
	//arma::unique(arma::round(arma::linspace(Urng[0], Urng[1], 20)/2/pi*ndir));
	//arma::mat kron_tmp1(arma::linspace<arma::rowvec>(0, 1, 20), arma::linspace<arma::rowvec>(0, 1, 20));
	arma::mat kron_tmp2(2, 1);

	for (int i=0; i < ndir; i++) {
		kron_tmp2(0) = cos(s_dir(i));
		kron_tmp2(1) = sin(s_dir(i));
		
		//arma::mat asdf = arma::unique(arma::round(arma::kron(Vrng*kron_tmp1, kron_tmp2)));
		//nextstate_sidxs.push_back(asdf);
		//std::cout << nextstate_sidxs[i] << std::endl;
		
	}
	



	QueryPerformanceFrequency(&Frequency);
	system("PAUSE");
	std::cout << pseq(1,0) << std::endl;

	//ptrGmap->printGlobalmap();

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
	arma::fmat pEst(2,H);
	for (int i = 0; i < H; i++) {
		pEst.col(i) = pseq.col(timestamps+i+1);
	}	
	std::cout << "pEst " << pEst << std::endl;
	std::vector<float> idxs_x = { std::fmin(arma::min(pEst.row(0)), pS[0] - Vrng[1] * T), std::fmax(arma::max(pEst.row(0)), pS[0] + Vrng[1] * T) };
	std::vector<float> idxs_y = { std::fmin(arma::min(pEst.row(1)), pS[1] - Vrng[1] * T), std::fmax(arma::max(pEst.row(1)), pS[1] + Vrng[1] * T) };

	float rlen = idxs_y.at(1) - idxs_y.at(0);
	float clen = idxs_x.at(1) - idxs_x.at(0);

	CCDPmap* ptrCCDP = new CCDPmap(ptrGmap, idxs_x, idxs_y, rlen, clen, H, 0.3);
	
	ptrCCDP->printLocalmap();
	//ptrCCDP->grid_resize(2);
	//ptrCCDP->printLocalmap();






	delete ptrCCDP;

}