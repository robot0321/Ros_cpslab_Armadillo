// Ros_cpslab_Armadillo.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"
#include <iostream>
#include <Windows.h>
#include <algorithm>
#include <cmath>
#include "CCDPmap.h"
#include <iomanip>

#ifndef ARMA_USE_CXX11
	#define ARMA_USE_CXX11
#endif

#define pi 3.141592


float Rs = 2; 
float ThetaS = 57.0 / 180.0 * pi;
float sensor_model[2] = { Rs, ThetaS };
int Nside = 4;
const float VMAX_const = 1.0; // It may need to be upper than Rs
float VMIN = -1.0;
arma::mat Urng(1, 2); 
arma::mat Vrng(1, 2); 
const int H = 5; // horizon
const float T = 1.0;
const float threshold = 1.36;
int ndir = 10;
int nvis = 20;
arma::vec pT_est;
arma::vec s;
float phi = 2 * pi / ndir;
arma::vec pS(3);
float grid;
float grid_reverse;
float Rfree;
float lagcoeff;
std::vector<float> lagrng;
float lag_feas = 1;
std::vector<float> Optctrl;
arma::mat mat_conv2 = arma::mat().ones(int(Rfree * 2 + 1), int(Rfree * 2 + 1));
float rlen;
float clen;

arma::mat pEst(2, H);
arma::rowvec idxs_x(2);
arma::rowvec idxs_y(2);
arma::mat s_dir(1, ndir);
arma::mat phi_a(Nside, 1);
arma::mat cc(Nside, 1);
arma::cube A(Nside, 2, ndir);
arma::mat mat_idxs_x;
arma::mat mat_idxs_y;
arma::mat shiftdir;
std::vector<arma::mat> nextstate_sidxs; // 2 x column vector
std::vector<arma::mat> nextstate_dir; // column vector



LARGE_INTEGER Frequency;
LARGE_INTEGER BeginTime;
LARGE_INTEGER Endtime;

arma::mat pseq;
Gmap* ptrGmap = new Gmap(1,true);
arma::rowvec gmap_xaxis;
arma::rowvec gmap_yaxis;
std::vector<arma::mat> Cost_visual;
std::vector<arma::mat> Cost_avoid;
std::vector<arma::cube> Cost_track;
std::vector<arma::cube> Cost_const; // avoid + const
CCDPmap* ptrCCDP;
std::vector<arma::cube> Optctrl_row;
std::vector<arma::cube> Optctrl_col;
std::vector<arma::cube> Optctrl_dirs;
std::vector<arma::cube> Jmin;
arma::cube Lk;
arma::mat Jmin_dir;
arma::mat mindiridx;
arma::mat currs_idx(3, H + 1);
arma::mat optidx(3, H);
arma::mat predict_s(3, H + 1);
float uv_opt;
float uw_opt;
std::vector<arma::vec> pSlog;
std::vector<arma::vec> pTlog;


void funf(int a);
arma::mat unique_rows(arma::mat A);
void initCostfunc(float grid);
void backRecursion(CCDPmap* ptr);
void mapPrint(bool obj);

void myPrint(const arma::mat M, bool obj=true) {
	for (int i = 0; i <M.n_rows; i++) {
		for (int j = 0; j < M.n_cols; j++) {
			if (obj && ptrCCDP->getLocalmap()(i, j) == 1) {
				if (M(i, j) == 1) { printf(" * "); }
				else { printf(" $ "); }
			}else {
				printf("%2.0f ", (int)M(i, j)); // = pow(pEst(0, tt) - (ptrCCDP->getGrid()*j + idxs_x[0]), 2) + pow(pEst(1, tt) - (ptrCCDP->getGrid()*i + idxs_y[0]), 2);
			}
		}
		printf("\n");
	}
	printf("\n\n");
}

void myPrint(const arma::umat M, bool obj = true) {
	for (int i = 0; i <M.n_rows; i++) {
		for (int j = 0; j < M.n_cols; j++) {
			if (obj && ptrCCDP->getLocalmap()(i, j) == 1) {
				if (M(i, j) == 1) { printf(" * "); }
				else { printf(" $ "); }
			}
			else {
				printf("%d ", M(i, j)); // = pow(pEst(0, tt) - (ptrCCDP->getGrid()*j + idxs_x[0]), 2) + pow(pEst(1, tt) - (ptrCCDP->getGrid()*i + idxs_y[0]), 2);
			}
		}
		printf("\n");
	}
	printf("\n\n");
}

void myPrint(const arma::cube Q, bool obj = true) {
	for (int k = 0; k < Q.n_slices; k++) {
		printf("slice %d \n", k);
		for (int i = 0; i < Q.n_rows; i++) {
			for (int j = 0; j < Q.n_cols; j++) {
				if (obj && ptrCCDP->getLocalmap()(i, j) == 1) {
					if (Q(i, j, k) == 1) { printf(" * "); }
					else { printf(" $ "); }
				}
				else {
					printf("%2.0f ", Q(i, j, k)); // = pow(pEst(0, tt) - (ptrCCDP->getGrid()*j + idxs_x[0]), 2) + pow(pEst(1, tt) - (ptrCCDP->getGrid()*i + idxs_y[0]), 2);
				}
			}
			printf("\n");
		}
		printf("\n\n");
	}
}



int main()
{	
	/*
	float a = 0.1;
	double aa = 0.1;
	long double aaa = 0.1;
	long double aaaa = 0.1;
	float b = 1.05;
	float bb = 0.95;
	float c = 10;
	float d = 1 / a;

	std::cout << roundf(b / a) << " " << int(b / a) << " " << roundf(b*c) << " " << int(b*c) << " " << roundf(b*d) << " " << int(b*d) << std::endl;
	std::cout << roundf(b / aa) << " " << roundf(b / aaa) << " " << roundf(b / aaaa) << std::endl;
	std::cout << int(b / aa) << " " << int(b / aaa) << " " << int(b / aaaa) << std::endl;
	std::cout << roundf(bb / a) << " " << int(bb / a) << " " << roundf(bb*c) << " " << int(bb*c) << " " << roundf(bb*d) << " " << int(bb*d) << std::endl;
	std::cout << roundf(bb / aa) << " " << roundf(bb / aaa) << " " << roundf(bb / aaaa) << std::endl;
	std::cout << int(bb / aa) << " " << int(bb / aaa) << " " << int(bb / aaaa) << std::endl;
	system("PAUSE");
	*/

	/*
	arma::mat a;
	a << 1 << 2 << 3 << arma::endr
	  << 4 << 0 << 0 << arma::endr
	  << 7 << 1 << 1 << arma::endr;
	std::cout << a << std::endl;
	arma::mat abc;
	abc = arma::trans(a);
	std::cout << abc << std::endl;
	std::cout << "a > abc" << std::endl << (a > abc) << std::endl;
	std::cout << "a && abc" << std::endl << (a && abc) << std::endl;
	std::cout << "a || abc" << std::endl << (a || abc) << std::endl;
	std::cout << "a == abc" << std::endl << (a == abc) << std::endl;

	arma::mat c(30,30);
	c.ones();
	for (int i = 0; i < 30; i++) {
		c.col(i).fill(i);
	}
	myPrint(c, false);
	arma::mat cc = arma::trans(c);
	myPrint(cc, false);
	arma::umat c5;
	c5 = c > cc;
	myPrint(c5, false);
	std::cout << (c5) << std::endl;

	arma::umat b;
	b << 0 << 0 << 1 << 1 << 1 << 2 << 2 << arma::endr
	  << 0 << 1 << 0 << 1 << 2 << 1 << 2 << arma::endr;
	std::cout << b << std::endl;

	arma::uword aa = arma::sub2ind(arma::size(a), 1,1);
	std::cout << a.submat(0,1,1,2) << std::endl;

	system("PAUSE");
	*/
	
	pseq << 0.000000 << 0.413207 << 0.822377 << 1.309345 << 1.685894 << 1.646640 << 1.296334 << 0.798910 << 0.346318 << -0.153285 << -0.600453 << -1.098636 << -1.508218 << -1.737180 << -1.571672 << -1.654645 << -2.041421 << -2.443007 << -2.887591 << -3.255841 << -3.634587 << -4.128173 << -4.540445 << -4.906420 << -4.875399 << -4.888977 << -4.472801 << -4.075093 << -3.577088 << -3.118782 << -2.631571 << -2.145281 << -1.702628 << -1.211647 << -0.725105 << -0.456884 << -0.641909 << -0.521413 << -0.119572 << 0.337383 << 0.832356 << 1.114022 << 1.462193 << 1.864227 << 2.340507 << 2.478401 << 2.938042 << 3.414545 << 3.546227 << 3.754268 << 3.599413 << 3.111832 << 2.613032 << 2.196940 << 1.991064 << 2.010829 << 1.983074 << 1.571468 << 1.087900 << 0.678270 << 0.605745 << 0.256468 << 0.068915 << 0.125748 << -0.258129 << -0.233119 << -0.134159 << -0.239640 << -0.705608 << -1.192281 << -1.417327 << -1.206516 << -0.787878 << -0.496964 << -0.109218 << 0.280981 << 0.246915 << 0.450310 << 0.841999 << 1.025085 << 1.230869 << 1.119091 << 0.799602 << 0.301458 << 0.028176 << 0.071591 << 0.055655 << 0.036528 << 0.120825 << 0.549576 << 0.997417 << 1.162366 << 1.605912 << 1.873580 << 1.952748 << 1.769193 << 1.341482 << 0.867441 << 0.385495 << 0.237447 << 0.029895 << 0.008932 << 0.288592 << 0.776588 << 1.270058 << 1.664245 << 2.147503 << 2.626513 << 3.083312 << 3.580669 << 4.048823 << 4.444888 << 4.501149 << 4.139665 << 3.641922 << 3.161207 << 2.772125 << 2.296023 << 1.871852 << 1.386357 << 0.935837 << 0.552073 << 0.607055 << 0.236032 << -0.263428 << -0.695211 << -1.193596 << -1.679028 << -2.006399 << -2.448779 << -2.820045 << -2.747362 << -2.784483 << -2.384064 << -1.896068 << -1.448597 << -0.962902 << -0.471652 << -0.169421 << -0.288938 << -0.044729 << -0.103854 << 0.245642 << 0.700924 << 1.132707 << 1.627851 << 2.059795 << 2.058974 << 1.816213 << 1.320090 << arma::endr
		 << 1.000000 << 0.718468 << 0.431101 << 0.544512 << 0.873466 << 1.371923 << 1.728695 << 1.678009 << 1.465498 << 1.445581 << 1.669278 << 1.711874 << 1.425094 << 0.980599 << 0.508787 << 0.015719 << -0.301146 << -0.599020 << -0.827812 << -1.166030 << -1.492452 << -1.412621 << -1.129721 << -0.789043 << -0.290006 << 0.209810 << 0.486933 << 0.789965 << 0.834585 << 0.634697 << 0.747060 << 0.630774 << 0.863279 << 0.768739 << 0.883963 << 1.305932 << 1.770438 << 2.255701 << 2.553230 << 2.756189 << 2.685470 << 2.272355 << 1.913499 << 1.616231 << 1.768406 << 2.249016 << 2.445815 << 2.294338 << 1.811990 << 1.357327 << 0.881911 << 0.771164 << 0.805791 << 1.083040 << 1.538688 << 2.038297 << 2.537526 << 2.821393 << 2.948523 << 2.661813 << 2.167101 << 1.809321 << 1.345830 << 0.849071 << 0.528699 << 0.029325 << -0.460785 << -0.949532 << -1.130842 << -1.016170 << -0.569679 << -0.116293 << 0.157097 << 0.563753 << 0.879431 << 1.192073 << 1.690911 << 2.147672 << 2.458445 << 2.923718 << 3.379408 << 3.866754 << 4.251366 << 4.208332 << 3.789623 << 3.291511 << 2.791765 << 2.292131 << 1.799288 << 1.542049 << 1.764397 << 2.236406 << 2.467203 << 2.889523 << 3.383215 << 3.848304 << 4.107269 << 4.266284 << 4.133140 << 3.655561 << 3.200674 << 2.701113 << 2.286638 << 2.177733 << 2.258280 << 2.565877 << 2.437573 << 2.294222 << 2.090913 << 2.142260 << 2.317850 << 2.623025 << 3.119850 << 3.465290 << 3.417829 << 3.555354 << 3.241323 << 3.394053 << 3.129328 << 3.009768 << 2.792901 << 2.472394 << 1.975426 << 1.640251 << 1.617025 << 1.364908 << 1.405072 << 1.285258 << 0.907332 << 0.674308 << 0.339404 << -0.155285 << -0.653905 << -0.953346 << -1.062251 << -0.839160 << -0.720415 << -0.627286 << -0.228969 << 0.256536 << 0.692841 << 1.189333 << 1.546897 << 1.753583 << 1.501466 << 1.431951 << 1.683792 << 2.183792 << 2.620904 << 2.683054 << arma::endr;
	
	grid = ptrGmap->getGrid();
	grid_reverse = 10;

	int ttmp = 0;
	Urng << -pi / 2.0 << pi / 2.0;
	Vrng << VMIN << VMAX_const;

	s << pseq(0, ttmp) - 0.5 << arma::endr
		<< pseq(1, ttmp) - 0.5 << arma::endr;// 0.5, -0.5
	phi = 2.0*pi / ndir;
	pS << s(0) << arma::endr
		<< s(1) << arma::endr
		<< phi << arma::endr;
	std::cout << pS << std::endl;

	for (int i = 0; i < ndir; i++) { s_dir(i) = 2 * PI*i / ndir; }

	for (int i = 0; i < Nside; i++) {//setting phi_a
		if (i == 0) { phi_a(0) = (ThetaS / 2 + pi / 2); cc(i) = 0; }
		else if (i == 1) { phi_a(1) = (-ThetaS / 2 - pi / 2); cc(i) = 0; }
		else { phi_a(i) = ThetaS*(2.0 * (i + 1) - 3 - Nside) / (2.0 * (Nside - 2)); cc(i) = Rs * cos(ThetaS / 2.0 / (Nside - 2)); }
	}

	//A << arma::cos(phi_a.col(0) + s_dir) << arma::sin(phi_a.col(0) + s_dir);
	for (int i = 0; i < ndir; i++) {
		A.slice(i).col(0) = cos(phi_a.col(0) + s_dir(i));
		A.slice(i).col(1) = sin(phi_a.col(0) + s_dir(i));
	}
	//std::cout << A << std::endl;

	shiftdir = arma::unique(arma::round(arma::linspace(Urng[0], Urng[1], 20) / 2.0 / pi*ndir));
	std::cout << shiftdir << std::endl;
	arma::mat kron_tmp1(2, 20);
	kron_tmp1.row(0) = arma::linspace<arma::rowvec>(0, 1, 20);
	kron_tmp1.row(1) = arma::linspace<arma::rowvec>(1, 0, 20);
	arma::mat kron_tmp2(2, 1);
	//std::cout << kron_tmp1 << std::endl;

	gmap_xaxis = ptrGmap->get_xaxis();
	gmap_yaxis = ptrGmap->get_yaxis();
	//std::cout << gmap_xaxis << std::endl;
	//std::cout << gmap_yaxis << std::endl;

	for (int i = 0; i < ndir; i++) {
		kron_tmp2(0) = sin(s_dir(i)) * grid_reverse;
		kron_tmp2(1) = cos(s_dir(i)) * grid_reverse;

		//std::cout << arma::trans(arma::round(arma::kron(Vrng*kron_tmp1, kron_tmp2))) << std::endl;

		nextstate_sidxs.push_back(unique_rows(arma::trans(arma::round(arma::kron(Vrng*kron_tmp1, kron_tmp2)))));
		

		arma::mat shiftdir_buffer(shiftdir);
		for (int j = 0; j < shiftdir.size(); j++) {
			shiftdir_buffer(j) = (int(shiftdir(j)) + i + ndir) % ndir;// -1;
																	  //if (shiftdir_buffer(j) == -1) shiftdir_buffer(j) = ndir -1;
		}
		nextstate_dir.push_back(shiftdir_buffer); // column vector
												  //std::cout << nextstate_sidxs[i] << std::endl;
											      //std::cout << shiftdir_buffer << std::endl;
												  //system("PAUSE");
	}

	QueryPerformanceFrequency(&Frequency);

	//std::cout << pseq(1,0) << std::endl;

	ptrGmap->printGlobalmap();
	//std::cout << std::fixed << std::setprecision(3) << std::setfill('0');	
	std::cout << arma::find(ptrGmap->getGlobalmap(), 1, "first") << std::endl;
	std::cout << arma::size(ptrGmap->getGlobalmap()) << std::endl;

	system("PAUSE");
	for (int timestamps = ttmp; timestamps < pseq.n_cols-H; timestamps++) {

		std::cout << "*************************** [ " << timestamps << " ] ***************************" << std::endl;
		
		QueryPerformanceCounter(&BeginTime);
		funf(timestamps);
		QueryPerformanceCounter(&Endtime);

		int64_t elapsed = Endtime.QuadPart - BeginTime.QuadPart;
		double duringtime = (double)elapsed / (double)Frequency.QuadPart;
		std::cout << "timestamps: " << timestamps << " takes " << duringtime << " seconds." << std::endl;
		std::cout << "********************************************************************************" << std::endl;
		//system("PAUSE");
	}
    return 0;
}

void funf(int timestamps) {
	pT_est << pseq(0, timestamps+1) << arma::endr
		<< pseq(1, timestamps+1) << arma::endr;
	pEst = pseq.cols(timestamps+1, timestamps + H);

	std::cout << "pEst " << pEst << std::endl;
	


	//idxs_x << gmap_xaxis >= arma::min(pEst.row(0)) - VMAX_const*T |  gmap_xaxis >= pS[0] - VMAX_const * T, ptrGmap->getXrng()[0]) << std::min(float(std::max(arma::max(pEst.row(0)), pS[0] + VMAX_const * T)), ptrGmap->getXrng()[1]);
	//idxs_y << std::max(float(std::min(arma::min(pEst.row(1)), pS[1] - VMAX_const * T)), ptrGmap->getYrng()[0]) << std::min(float(std::max(arma::max(pEst.row(1)), pS[1] + VMAX_const * T)), ptrGmap->getYrng()[1]); 
	
	arma::uvec q1 = arma::find(gmap_xaxis >= arma::min(pEst.row(0)) - VMAX_const*T, 1, "first");
	arma::uvec q2 = arma::find(gmap_xaxis >= pS[0] - VMAX_const * T, 1, "first");
	idxs_x(0) = std::min(q1(0), q2(0));
	q1 = arma::find(gmap_xaxis <= arma::max(pEst.row(0)) + VMAX_const*T, 1, "last");
	q2 = arma::find(gmap_xaxis <= pS[0] + VMAX_const * T, 1, "last");
	idxs_x(1) = std::max(q1(0), q2(0));
	q1 = arma::find(gmap_yaxis >= arma::min(pEst.row(1)) - VMAX_const*T, 1, "first");
	q2 = arma::find(gmap_yaxis >= pS[1] - VMAX_const * T, 1, "first");
	idxs_y(0) = std::min(q1(0), q2(0));
	q1 = arma::find(gmap_yaxis <= arma::max(pEst.row(1)) + VMAX_const*T, 1, "last");
	q2 = arma::find(gmap_yaxis <= pS[1] + VMAX_const * T, 1, "last");
	idxs_y(1) = std::max(q1(0), q2(0));

	std::cout << idxs_x <<" " << idxs_y << std::endl;
	//std::cout << gmap_xaxis.subvec(idxs_x(0), idxs_x(1)) << std::endl;
	//std::cout << gmap_yaxis.subvec(idxs_y(0), idxs_y(1)) << std::endl;

	rlen = idxs_y(1) - idxs_y(0);
	clen = idxs_x(1) - idxs_x(0);

	std::cout << "idxs_x: " << gmap_xaxis(idxs_x(0)) << " / " << gmap_xaxis(idxs_x(1)) << std::endl;
	std::cout << "idxs_y: " << gmap_yaxis(idxs_y(0)) << " / " << gmap_yaxis(idxs_y(1)) << std::endl;
	//system("PAUSE");

	ptrCCDP = new CCDPmap(ptrGmap, idxs_x, idxs_y, threshold);
	//ptrCCDP->printLocalmap();
	std::cout << "CCDP size : " << ptrCCDP->getLocal_num_y() << " x " << ptrCCDP->getLocal_num_x() << std::endl;
	grid = ptrCCDP->getGrid();
	mat_idxs_x = arma::mat(ptrCCDP->getLocalmap());
	mat_idxs_y = arma::mat(ptrCCDP->getLocalmap());
	for (int i = 0; i < mat_idxs_x.n_cols; i++) { mat_idxs_x.col(i).fill(gmap_xaxis(idxs_x(0)) + i * grid); }
	for (int i = 0; i < mat_idxs_y.n_rows; i++) { mat_idxs_y.row(i).fill(gmap_yaxis(idxs_y(0)) + i * grid); }
	//std::cout << "matX" << mat_idxs_x << std::endl;
	//std::cout << "matY" << mat_idxs_y << std::endl;
	

	//std::cin.get();
	//system("PAUSE");

	initCostfunc(grid);
	backRecursion(ptrCCDP);

	Cost_avoid.clear();
	Cost_const.clear();
	Cost_track.clear();
	Cost_visual.clear();
	Optctrl_col.clear();
	Optctrl_row.clear();
	Optctrl_dirs.clear();
	Jmin.clear();



	delete ptrCCDP;

}

arma::mat unique_rows(arma::mat A) {
	std::vector<arma::mat> list_of_mat;
	std::vector<arma::mat> sorted_list;
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

	//sort
	int count;
	for (int i = 0; i < list_of_mat.size(); i++) {
		if (i == 0) { 
			sorted_list.push_back(list_of_mat[i]); 
		}else {
			count = 0;
			for (int j = 0; j < sorted_list.size(); j++) {
				
				if (sorted_list[j](0) == list_of_mat[i](0)) {
					if (sorted_list[j](1) < list_of_mat[i](1)) {
						sorted_list.insert(sorted_list.begin() + j, list_of_mat[i]);
						break;
					}
				}
				else if (sorted_list[j](0) < list_of_mat[i](0)) {
					sorted_list.insert(sorted_list.begin() + j, list_of_mat[i]);
					break;
				}
				count++;
			}
			if (count == sorted_list.size()) { sorted_list.push_back(list_of_mat[i]); }
		}

	}

	//for return matrix
	arma::mat Ret(sorted_list.size(), 2);
	for (int i = 0; i < sorted_list.size(); i++) {
		Ret.row(i) = sorted_list[i];
	}
	list_of_mat.clear();
	sorted_list.clear();

	return Ret;
}





void initCostfunc(float grid) {
	arma::mat local = ptrCCDP->getLocalmap();
	//std::cout << "local" << std::endl;
	//myPrint(local);
	for (int i = 0; i < H + 1; i++) {
		Cost_visual.push_back(arma::zeros<arma::mat>(local.n_rows, local.n_cols));
		Cost_avoid.push_back(arma::zeros<arma::mat>(local.n_rows, local.n_cols));
		Cost_track.push_back(arma::zeros<arma::cube>(local.n_rows, local.n_cols, ndir));
		Cost_const.push_back(arma::zeros<arma::cube>(local.n_rows, local.n_cols, ndir));
	}
	arma::mat dist2target(local);
	arma::mat angle2target(local);
	arma::mat minradius = arma::zeros<arma::mat>(1, nvis);
	arma::mat cvistmp(local);

	arma::mat tmp(local);

	for (int tt = 0; tt < H; tt++) {
		/****************************** [ Cvis ] *******************************/
		arma::mat ccdp_ys = mat_idxs_y - pEst(1, tt);
		arma::mat ccdp_xs = mat_idxs_x - pEst(0, tt);

		dist2target = arma::pow(ccdp_xs, 2) + arma::pow(ccdp_ys, 2);
		//std::cout << dist2target << std::endl;
		//system("PAUSE");
		angle2target = round((arma::atan2(ccdp_ys, ccdp_xs) + pi) / 2.0 / pi*nvis) + 1;
		angle2target.elem(arma::find(angle2target == nvis + 1)).fill(1);
		//myPrint(angle2target, false);
		//std::cout << angle2target << std::endl;

		minradius.zeros();
		cvistmp.zeros();
		for (int ii = 0; ii < nvis; ii++) {
			arma::uvec obsidx = arma::find(local >= 1 && angle2target == ii);
			if (arma::sum(obsidx) >= 1) {
				minradius(ii) = arma::min(dist2target(obsidx));
				cvistmp.elem(arma::find(dist2target >= minradius(ii) && angle2target == ii)).ones();
			}
		}
		Cost_visual[tt + 1] = cvistmp;

		//std::cout << "tt: " << tt << " / Cvis : " << std::endl;
		//std::cout << "cost_visual" << std::endl;
		//myPrint(Cost_visual[tt + 1],false);
		//std::cout << cvistmp << std::endl;
		//system("PAUSE");


		/****************************** [ Ctrack ] *******************************/
		arma::mat afterA(local.n_elem, 2);
		afterA.col(0) = arma::vectorise(ccdp_xs);
		afterA.col(1) = arma::vectorise(ccdp_ys);
		//std::cout << afterA << std::endl; //
		arma::mat B;
		arma::mat C(local);
		for (int ii = 0; ii < ndir; ii++) {
			B = (A.slice(ii)*arma::trans(afterA));
			B.each_col() += cc;
			C.zeros();
			C = arma::sum(1 - 0.5 * erfc(B / (-sqrt(2))), 0);
			//C.elem(arma::find(arma::sum(1 - 0.5 * erfc(B / (-sqrt(2))), 0)>threshold)).ones();
			Cost_track[tt + 1].slice(ii) = arma::reshape(C, local.n_rows, local.n_cols);
			//std::cout << "slice "<<ii<<std::endl << Cost_track[tt + 1].slice(ii) << std::endl;
			//##########################################################################			
			//myPrint(Cost_track[tt + 1].slice(ii));
			//std::cout << arma::sum(1 - 0.5 * erfc(B / (-sqrt(2))), 0) << std::endl;
			//std::cin.get();
		}
		//myPrint(Cost_track[tt + 1]);
		//std::cout << Cost_track[tt + 1] << std::endl;
		//system("PAUSE");

		/****************************** [ Cavoid ] *******************************/
		//Cost_avoid[tt + 1] = arma::ceil(arma::conv2(local, mat_conv2, "same"));
		Cost_avoid[tt + 1] = arma::mat(local);
		for (int i = 0; i < local.n_rows; i++) {
			for (int j = 0; j < local.n_cols; j++) {
				Cost_avoid[tt + 1](i, j) = 4*(local(i, j) || Cost_visual[tt + 1](i, j));
			}
		}
		//std::cout << "#######################################################################################################################################" << std::endl;
		//std::cout << "cost avoid" << std::endl;
		//myPrint(Cost_avoid[tt + 1],false);
		//std::cout << Cost_avoid[tt + 1] << std::endl;
		//system("PAUSE");

		/**************************	*** [ Cconst ] *******************************/
		Cost_const[tt + 1] = Cost_track[tt + 1].each_slice() + Cost_avoid[tt + 1]; 
		//myPrint(Cost_const[tt + 1], false);
		//std::cout << Cost_const[tt+1] << std::endl;

		
	}

}


void backRecursion(CCDPmap* ptr) {
	arma::mat local = ptr->getLocalmap();
	//float grid = ptr->getGrid();


	// -----[Backward recursion] ----- 
	// find the optimal control when lambda is given


	//std::vector<arma::mat> CompMat;

	for (int i = 0; i < H; i++) {
		Optctrl_row.push_back(arma::cube(local.n_rows, local.n_cols, ndir));
		Optctrl_col.push_back(arma::cube(local.n_rows, local.n_cols, ndir));
		Optctrl_dirs.push_back(arma::cube(local.n_rows, local.n_cols, ndir));
	}
	for (int i = 0; i < H + 1; i++) { Jmin.push_back(arma::cube(local.n_rows, local.n_cols, ndir).fill(100)); }

	Lk = arma::cube(local.n_rows, local.n_cols, ndir);
	Jmin_dir = arma::mat(local).fill(100);
	mindiridx = arma::mat(local).fill(0);
	//for (int i = 0; i < H + 1; i++) { CompMat.push_back(arma::mat()); }

	lagcoeff = 0;
	lagrng = { 0.0, 1.0 };
	lag_feas = 1;

	for (int ilagran = 0; ilagran < 100; ilagran++) {
		arma::mat mat_lagcoeff = arma::mat(local.n_rows, local.n_cols).fill(lagcoeff);
		for (int i = 0; i < H; i++) {
			Optctrl_row[i].zeros();
			Optctrl_col[i].zeros();
			Optctrl_dirs[i].zeros();
		}
		for (int i = 0; i < H + 1; i++) { Jmin[i].fill(100); }

		//std::cout << "Jmin[H]" << std::endl;
		Jmin[H + 1 - 1] = (Cost_track[H + 1 - 1].each_slice() % mat_lagcoeff).each_slice() + Cost_avoid[H + 1 - 1];
		//myPrint(Jmin[H], false);
		//system("PAUSE");
		//std::cout << Jmin[H] << std::endl;

		for (int tt = H - 1; tt >= 0; tt--) {
			//Lk = (Cost_const[tt].each_slice() % mat_lagcoeff).each_slice() + lag_feas * Cost_visual[tt];
			Lk = (Cost_track[tt].each_slice() % mat_lagcoeff).each_slice() + Cost_avoid[tt];
			//myPrint(Lk, false);
			//system("PAUSE");
			for (int ii = 0; ii < ndir; ii++) {
				Jmin_dir = arma::mat(local).fill(100);	
				for (int j = 0; j < local.n_rows; j++) {
					for (int k = 0; k < local.n_cols; k++) {
						for (int i = 0; i <	 nextstate_dir[ii].n_elem; i++) { //assume that nextstate_dir is rowvec or colvec
							if (Jmin_dir(j, k) > Jmin[tt + 1](j, k, nextstate_dir[ii](i))) {
								Jmin_dir(j, k) = Jmin[tt + 1](j, k, nextstate_dir[ii](i));
								mindiridx(j, k) = nextstate_dir[ii](i);
							}
						}
					}
				}
				//myPrint(Jmin_dir, false);
				//myPrint(mindiridx, false);
				//system("PAUSE");


				for (int jj = 0; jj < nextstate_sidxs[ii].n_rows; jj++) {
					std::vector<int> original_rng_r = { std::max(0,int(0 - nextstate_sidxs[ii](jj,0))), std::min(int(local.n_rows - 1), int(local.n_rows - 1 - nextstate_sidxs[ii](jj,0))) };
					std::vector<int> original_rng_c = { std::max(0,int(0 - nextstate_sidxs[ii](jj,1))), std::min(int(local.n_cols - 1), int(local.n_cols - 1 - nextstate_sidxs[ii](jj,1))) };
					std::vector<int> comp_rng_r = { std::max(0,int(0 + nextstate_sidxs[ii](jj,0))), std::min(int(local.n_rows - 1), int(local.n_rows - 1 + nextstate_sidxs[ii](jj,0))) };
					std::vector<int> comp_rng_c = { std::max(0,int(0 + nextstate_sidxs[ii](jj,1))), std::min(int(local.n_cols - 1), int(local.n_cols - 1 + nextstate_sidxs[ii](jj,1))) };

					/*
					std::cout << "jj is " << jj << std::endl;
					s1td::cout << original_rng_r[0] << " " << original_rng_r[1] << std::endl;
					std::cout << original_rng_c[0] << " " << original_rng_c[1] << std::endl;
					std::cout << comp_rng_r[0] << " " << comp_rng_r[1] << std::endl;
					std::cout << comp_rng_c[0] << " " << comp_rng_c[1] << std::endl;
					*/

					//arma::urowvec vec_original_rng_r = arma::linspace<arma::urowvec>(original_rng_r[0], original_rng_r[1], original_rng_r[1] - original_rng_r[0] + 1);
					//arma::urowvec vec_original_rng_c = arma::linspace<arma::urowvec>(original_rng_c[0], original_rng_c[1], original_rng_c[1] - original_rng_c[0] + 1);
					//arma::urowvec vec_comp_rng_r = arma::linspace<arma::urowvec>(comp_rng_r[0], comp_rng_r[1], comp_rng_r[1] - comp_rng_r[0] + 1);
					//arma::urowvec vec_comp_rng_c = arma::linspace<arma::urowvec>(comp_rng_c[0], comp_rng_c[1], comp_rng_c[1] - comp_rng_c[0] + 1);
					//std::cout << vec_original_rng_r << std::endl;

					arma::umat CompMat;
					/*
					arma::umat CompMat1;
					arma::umat AA;
					arma::umat A;
					arma::mat A1;
					arma::mat A2;
					arma::umat B;
					arma::mat B1;
					arma::mat B2;
					arma::umat C;
					*/
					//AA = Jmin[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) >= Jmin_dir.submat(comp_rng_r[0], comp_rng_c[0], comp_rng_r[1], comp_rng_c[1]);
					/*
					A = Jmin[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) > Jmin_dir.submat(comp_rng_r[0], comp_rng_c[0], comp_rng_r[1], comp_rng_c[1]);
					B = Jmin[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) == Jmin_dir.submat(comp_rng_r[0], comp_rng_c[0], comp_rng_r[1], comp_rng_c[1]);
					B1 = Jmin[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]);
					B2 = Jmin_dir.submat(comp_rng_r[0], comp_rng_c[0], comp_rng_r[1], comp_rng_c[1]);
					C = abs(Optctrl_dirs[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) > abs(mindiridx.submat(comp_rng_r[0], comp_rng_c[0], comp_rng_r[1], comp_rng_c[1])));
					CompMat1 = A || B && C;
					std::cout << "A" << std::endl;
					myPrint(A, false);
					std::cout << "B" << std::endl;
					myPrint(B, false);
					std::cout << "B1" << std::endl;
					myPrint(B1, false);
					std::cout << "B2" << std::endl;
					myPrint(B2, false);
					std::cout << "B1>B2 " << B1.size() << " " << B2.size() << std::endl;			
					myPrint((B1>B2), false);
					std::cout << "C" << std::endl;
					myPrint(C, false);
					std::cout << "CompMat1" << std::endl;
					myPrint(CompMat1, false);
					*/

					CompMat = Jmin[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) > Jmin_dir.submat(comp_rng_r[0], comp_rng_c[0], comp_rng_r[1], comp_rng_c[1]) || (Jmin[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) == Jmin_dir.submat(comp_rng_r[0], comp_rng_c[0], comp_rng_r[1], comp_rng_c[1]) && abs(Optctrl_dirs[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) > abs(mindiridx.submat(comp_rng_r[0], comp_rng_c[0], comp_rng_r[1], comp_rng_c[1]))));
					//myPrint(CompMat, false);
					//system("PAUSE");

					//std::cout << CompMat << std::endl;

					/*
					Jmin[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) = (1 - CompMat) % Jmin[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) + CompMat % Jmin_dir.submat(comp_rng_r[0], comp_rng_c[0], comp_rng_r[1], comp_rng_c[1]);
					Optctrl_row[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) = (1 - CompMat) % Optctrl_row[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) + CompMat * nextstate_sidxs[ii](jj, 0);
					Optctrl_col[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) = (1 - CompMat) % Optctrl_col[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) + CompMat * nextstate_sidxs[ii](jj, 1);
					Optctrl_dirs[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) = (1 - CompMat) % Optctrl_dirs[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) + CompMat % mindiridx.submat(comp_rng_r[0], comp_rng_c[0], comp_rng_r[1], comp_rng_c[1]);
					*/


					/**/

					for (int i = 0; i < CompMat.n_rows; i++) {
						for (int j = 0; j < CompMat.n_cols; j++) {
							Jmin[tt](i + original_rng_r[0], j + original_rng_c[0], ii) = (1 - CompMat(i, j))*Jmin[tt](i + original_rng_r[0], j + original_rng_c[0], ii) + CompMat(i, j)*Jmin_dir(i + comp_rng_r[0], j + comp_rng_c[0]);
							Optctrl_row[tt](i + original_rng_r[0], j + original_rng_c[0], ii) = (1 - CompMat(i, j)) * Optctrl_row[tt](i + original_rng_r[0], j + original_rng_c[0], ii) + CompMat(i, j) * nextstate_sidxs[ii](jj, 0);
							Optctrl_col[tt](i + original_rng_r[0], j + original_rng_c[0], ii) = (1 - CompMat(i, j)) * Optctrl_col[tt](i + original_rng_r[0], j + original_rng_c[0], ii) + CompMat(i, j) * nextstate_sidxs[ii](jj, 1);
							Optctrl_dirs[tt](i + original_rng_r[0], j + original_rng_c[0], ii) = (1 - CompMat(i, j)) * Optctrl_dirs[tt](i + original_rng_r[0], j + original_rng_c[0], ii) + CompMat(i, j) * mindiridx(i + comp_rng_r[0], j + comp_rng_c[0]);
						}
					}

					Jmin[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) = (1 - CompMat) % Jmin[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) + CompMat % Jmin_dir.submat(comp_rng_r[0], comp_rng_c[0], comp_rng_r[1], comp_rng_c[1]);
					Optctrl_row[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) = (1 - CompMat) % Optctrl_row[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) + CompMat % arma::mat(arma::size(CompMat)).fill(nextstate_sidxs[ii](jj, 0));
					Optctrl_col[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) = (1 - CompMat) % Optctrl_col[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) + CompMat % arma::mat(arma::size(CompMat)).fill(nextstate_sidxs[ii](jj, 1));
					Optctrl_dirs[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) = (1 - CompMat) % Optctrl_dirs[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) + CompMat % mindiridx.submat(comp_rng_r[0], comp_rng_c[0], comp_rng_r[1], comp_rng_c[1]);

					
					//std::cout << arma::sum(arma::sum(Jmin[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) != JJ, 1)) << "/";
					//std::cout << arma::sum(arma::sum(Optctrl_row[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) != RR, 1))<<"/";
					//std::cout << arma::sum(arma::sum(Optctrl_col[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) != CC, 1))<<"/";
					//std::cout << arma::sum(arma::sum(Optctrl_dirs[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) != DD, 1)) << std::endl;

					//std::cout << Optctrl_col[tt].slice(ii).submat(original_rng_r[0], original_rng_c[0], original_rng_r[1], original_rng_c[1]) << std::endl;
					//std::cout << CC << std::endl;

					//std::cout << CompMat << std::endl;
					//myPrint(Optctrl_row[tt], false);
					//myPrint(Optctrl_col[tt], false);
					//myPrint(Optctrl_dirs[tt], false);
					//std::cout << "or/oc/cr/cc => /" << original_rng_r[0] << " " << original_rng_r[1] << "/" << original_rng_c[0] << " " << original_rng_c[1] << "/" << comp_rng_r[0] << " " << comp_rng_r[1] << "/" << comp_rng_c[0] << " " << comp_rng_c[1] << "/" << std::endl;
					//std::cout << "tt: " << tt << " / ii: " << ii << " / jj: " << jj << std::endl;
					//std::cout << "size of CompMat: " << CompMat.n_rows << " x " << CompMat.n_cols << std::endl;
					//myPrint(CompMat, false);
					//system("PAUSE");

				}

				//std::cout << nextstate_dir[ii].n_elem << std::endl;
				// for jj 1:size...

				//myPrint(Jmin_dir,false);
				//myPrint(mindiridx, false);
				//std::cout << "" << std::endl;
				//system("PAUSE");

			}
			Jmin[tt] = Jmin[tt] + Lk;
			//std::cout << Jmin[tt] << std::endl;
			// 
			//system("PAUSE");
		}

		//std::cout << std::round(std::round((s[0] - gmap_xaxis(idxs_x[0])) / grid/grid)*grid) << std::endl;
		//std::cout << std::round(std::round((s[1] - gmap_yaxis(idxs_y[0])) / grid/grid)*grid) << std::endl;


		currs_idx(0, 0) = roundf((s[1] - gmap_yaxis(idxs_y[0])) * grid_reverse );
		currs_idx(1, 0) = roundf((s[0] - gmap_xaxis(idxs_x[0])) * grid_reverse );
		currs_idx(2, 0) = int(round(pS[2] / 2.0 / pi*ndir)) % ndir;// - 1;
																   //if (currs_idx(2, 0) == -1) currs_idx(2, 0) = ndir - 1

		std::cout << currs_idx.col(0) << std::endl;


		float Costconst_val = 0;
		optidx.zeros();
		for (int tt = 0; tt < H; tt++) {
			optidx(0, tt) = Optctrl_row[tt](currs_idx(0, tt), currs_idx(1, tt), currs_idx(2, tt));
			optidx(1, tt) = Optctrl_col[tt](currs_idx(0, tt), currs_idx(1, tt), currs_idx(2, tt));
			optidx(2, tt) = Optctrl_dirs[tt](currs_idx(0, tt), currs_idx(1, tt), currs_idx(2, tt));
			currs_idx(0, tt + 1) = currs_idx(0, tt) + optidx(0, tt) ;
			currs_idx(1, tt + 1) = currs_idx(1, tt) + optidx(1, tt) ;
			currs_idx(2, tt + 1) = optidx(2, tt);


			//std::cout << optidx.col(tt) << std::endl;
			//std::cout << currs_idx.col(tt + 1) << std::endl;
			/*
			if (tt == H - 2) {
				std::cout << "row" << std::endl;
				myPrint(Optctrl_row[tt], false);
				std::cout << "col" << std::endl;
				myPrint(Optctrl_col[tt], false);
				std::cout << "dirs" << std::endl;
				myPrint(Optctrl_dirs[tt], false);
			}*/

			Costconst_val = Costconst_val + Cost_const[tt + 1](currs_idx(0, tt + 1), currs_idx(1, tt + 1), currs_idx(2, tt + 1));
			
		}
		std::cout << "Costconst_val: " << Costconst_val << std::endl;

		for (int i = 0; i < H + 1; i++) {
			predict_s(0, i) = (currs_idx(1, i))*grid + gmap_xaxis(idxs_x[0]);
			predict_s(1, i) = (currs_idx(0, i))*grid + gmap_yaxis(idxs_y[0]);
			predict_s(2, i) = (currs_idx(2, i))*2.0*pi / ndir; //
			//std::cout << predict_s.col(i) << std::endl;
		}

		if (Costconst_val <= 0 && ilagran == 0) {
			std::cout << "feasible //  " << ilagran+1 << " times" << std::endl;
			break;
		}
		else if (ilagran == 0) {
			lag_feas = 0;
			lagcoeff = 1;
		}

		if (ilagran == 1 && Costconst_val > 0) {
			//std::cout << "infeasible //  " 
			std::cout << ilagran+1 << " times" << std::endl;
			break;
		}
		else if (ilagran == 1) {
			lagcoeff = lagrng[1];
			lag_feas = 1;
		}
		if (ilagran > 1) {
			if (Costconst_val > 0) {
				lagrng[0] = lagcoeff;
			}
			else {
				lagrng[1] = lagcoeff;
			}
			if (lagrng[1] - lagrng[0] < 0.01) {
				std::cout << ilagran << " times / ";
				std::cout << lagrng[1] - lagrng[0] << std::endl;
				break;
			}
			else {
				lagcoeff = (lagrng[1] + lagrng[0]) / 2.0;
			}
		}
		
	}
	uv_opt = sqrt(optidx(0, 0)*optidx(0, 0) + optidx(1, 0)*optidx(1, 0))*grid / T;
	uw_opt = (predict_s(2, 1) - predict_s(2, 0)) / T;
	std::cout << uv_opt << std::endl;
	std::cout << uw_opt << std::endl;


	pS[0] = s[0] + cos(phi)*uv_opt*T;
	pS[1] = s[1] + sin(phi)*uv_opt*T;
	std::cout << "s0, s1" << s[0] << ", " << s[1] << std::endl;
	pS[2] = phi + uw_opt*T;
	for (int i = 0; i < 2; i++) { s[i] = pS[i]; }
	phi = pS[2];
	pSlog.push_back(pS);
	pTlog.push_back(pT_est);
	//std::cout << std::endl;
	std::cout << "  V: " << uv_opt << "  U: " << uw_opt << std::endl;
	std::cout << "human position: (" << pT_est[0] << ", " << pT_est[1] << ")" << std::endl;
	std::cout << "robot position: (" << pS[0] << ", " << pS[1] << ", " << round(pS[2] * ndir / (2 * pi)) *2*pi/ndir << ")" << std::endl;
	//myPrint(Cost_track[1].slice(currs_idx(2, 0)));

}

void mapPrint(bool obj = true) {
	arma::mat map = ptrGmap->getGlobalmap();
	arma::mat buffer_map(map);
	buffer_map.zeros();
	float grid = ptrGmap->getGrid();
	arma::mat avoid = arma::ceil(arma::conv2(map, mat_conv2, "same"));
	myPrint(avoid, false);
	myPrint(Cost_track[1].slice(currs_idx(2, 0)), false);

	for (int i = map.n_rows - 1; i >= 0; i--) {
		for (int j = 0; j < map.n_cols; j++) {
			if (j == int((round(pT_est[0]) - ptrGmap->getXrng()[0]) * grid_reverse) && i == int((round(pT_est[1]) - ptrGmap->getYrng()[0]) * grid_reverse)) {
				printf(" H ");
			}
			else if (j == int((round(s[0]) - ptrGmap->getXrng()[0]) * grid_reverse) && i == int((round(s[1]) - ptrGmap->getYrng()[0]) * grid_reverse)) {
				printf(" R ");
			}
			else if (j >= ((idxs_x[0] - int(ptrGmap->getXrng()[0])) * grid_reverse) && j < abs(idxs_x[1] - int(ptrGmap->getXrng()[0])) * grid_reverse && i >= (idxs_y[0] - int(ptrGmap->getYrng()[0])) * grid_reverse && i < (idxs_y[1] - int(ptrGmap->getYrng()[0])) * grid_reverse) {
				if (Cost_track[1].slice(currs_idx(2, 0))(i - int((round(idxs_y[0]) - (ptrGmap->getYrng()[0])) * grid_reverse), j - int((round(idxs_x[0]) - (ptrGmap->getXrng()[0])) * grid_reverse)) == 0) {
					printf(" Z ");
				}
				else if (avoid(i, j) > 0) {
					if (map(i, j) == 1) {
						printf(" $ ");
					}
					else {
						printf(" + ");
					}
				}
				else {
					printf(" . ");
				}
			}
			else if (avoid(i, j) > 0) {
				if (map(i, j) == 1) {
					printf(" $ ");
				}
				else {
					printf(" + ");
				}
			}
			else {
				printf(" . ");
			}

			/*
			if (obj && ptrCCDP->getLocalmap()(i, j) > 0) {
			if (M(i, j) > 0) { printf(" * "); }
			else { printf(" $ "); }
			}
			else {
			printf("%2.0f ", M(i, j)); // = pow(pEst(0, tt) - (ptrCCDP->getGrid()*j + idxs_x[0]), 2) + pow(pEst(1, tt) - (ptrCCDP->getGrid()*i + idxs_y[0]), 2);
			}
			*/
		}
		printf("\n");
	}
	printf("\n\n");
}

