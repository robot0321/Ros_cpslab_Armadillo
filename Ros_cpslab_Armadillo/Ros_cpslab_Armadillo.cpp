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


float Rs = 3; 
float ThetaS = 57.0 / 180.0 * pi;
float sensor_model[2] = { Rs, ThetaS };
int Nside = 4;
float VMAX_const = 4;
float VMIN = -4;
arma::mat Urng(1, 2); 
arma::mat Vrng(1, 2); 
const int H = 3; // horizon
const int T = 1;
const float threshold = 0.7;
int ndir = 10;
int nvis = 20;
arma::vec pT_est;
arma::vec s;
float phi = 2 * pi / ndir * 3;
arma::vec pS(3);
float grid;
float Rfree;
float lagcoeff;
std::vector<float> lagrng;
float lag_feas = 1;
std::vector<float> Optctrl;
arma::mat mat_conv2 = arma::mat().ones(int(Rfree * 2 + 1), int(Rfree * 2 + 1));
float rlen;
float clen;

arma::mat pEst(2, H);
arma::rowvec idxs_x;
arma::rowvec idxs_y;
arma::mat s_dir(1, ndir);
arma::mat phi_a(Nside, 1);
arma::mat cc(Nside, 1);
arma::cube A(Nside, 2, ndir);
arma::mat mat_idxs_x;
arma::mat mat_idxs_y;
arma::mat shiftdir;
std::vector<arma::mat> nextstate_sidxs;
std::vector<arma::mat> nextstate_dir;



LARGE_INTEGER Frequency;
LARGE_INTEGER BeginTime;
LARGE_INTEGER Endtime;

arma::mat pseq;
Gmap* ptrGmap = new Gmap();
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
				printf("%2.0f ", M(i, j)); // = pow(pEst(0, tt) - (ptrCCDP->getGrid()*j + idxs_x[0]), 2) + pow(pEst(1, tt) - (ptrCCDP->getGrid()*i + idxs_y[0]), 2);
			}
		}
		printf("\n");
	}
	printf("\n\n");
}



int main()
{	
	pseq << 0 << -1 << -1 << -2 << -3 << -4 << -5 << -6 << -7 << -7 << -7 << -7 << -7 << -7 << -7 << -6 << -5 << -4 << -3 << -2 << -1 << 0 << arma::endr
		 << 0 << 0  << 1  << 1  << 1  <<  1 << 1  <<  1 <<  1 <<  2 <<  3 <<  4 <<  5 <<  6 <<  7 <<  7 <<  7 <<  7 <<  7 <<  7 <<  7 << 7 << arma::endr;
	int ttmp = 0;
	Urng << -pi/2.0 << pi/2.0;
	Vrng << VMIN << VMAX_const;
	
	s << pseq(0, ttmp) - 1 << arma::endr
	  << pseq(1, ttmp) - 1 << arma::endr;// -1, -1
	phi = 2.0*pi/ndir*3;
	pS << s(0) << arma::endr
		<< s(1) << arma::endr
		<< phi << arma::endr;
	//std::cout << pS << std::endl;

	for (int i = 0; i < ndir; i++) { s_dir(i) = 2 * PI*i / ndir; }

	for (int i = 0; i < Nside; i++) {//setting phi_a
		if (i == 0) { phi_a(0) = (ThetaS / 2 + pi / 2); cc(i) = 0; }
		else if (i == 1) { phi_a(1) = (-ThetaS / 2 - pi / 2); cc(i) = 0; }
		else { phi_a(i) = ThetaS*(2.0 * (i+1) - 3 - Nside) / (2.0 * (Nside - 2)); cc(i) = Rs * cos(ThetaS / 2.0 / (Nside - 2)); }
	}
	
	//A << arma::cos(phi_a.col(0) + s_dir) << arma::sin(phi_a.col(0) + s_dir);
	for (int i = 0; i < ndir; i++) {
		A.slice(i).col(0) = cos(phi_a.col(0) + s_dir(i));
		A.slice(i).col(1) = sin(phi_a.col(0) + s_dir(i));
	}
	//std::cout << A << std::endl;

	shiftdir = arma::unique(arma::round(arma::linspace(Urng[0], Urng[1], 20)/2.0/pi*ndir));
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
			shiftdir_buffer(j) = (int(shiftdir(j)) + i + ndir) % ndir -1;
			if (shiftdir_buffer(j) == -1) shiftdir_buffer(j) = ndir -1;
		}
		nextstate_dir.push_back(shiftdir_buffer);
		//std::cout << nextstate_sidxs[i] << std::endl;
		//std::cout << shiftdir_buffer << std::endl;
	}

	QueryPerformanceFrequency(&Frequency);
	system("PAUSE");
	//std::cout << pseq(1,0) << std::endl;

	ptrGmap->printGlobalmap();
	//std::cout << std::fixed << std::setprecision(3) << std::setfill('0');

	

	system("PAUSE");
	for (int timestamps = 0; timestamps < pseq.n_cols-H; timestamps++) {

		std::cout << "*************************** [ " << timestamps << " ] ***************************" << std::endl;
		
		QueryPerformanceCounter(&BeginTime);
		funf(timestamps);
		QueryPerformanceCounter(&Endtime);

		int64_t elapsed = Endtime.QuadPart - BeginTime.QuadPart;
		double duringtime = (double)elapsed / (double)Frequency.QuadPart;
		std::cout << "timestamps: " << timestamps << " takes " << duringtime << " seconds." << std::endl;
		std::cout << "********************************************************************************" << std::endl;
		system("PAUSE");
	}
    return 0;
}

void funf(int timestamps) {
	pT_est << pseq(0, timestamps) << arma::endr
		   << pseq(1, timestamps) << arma::endr;
	pEst = pseq.cols(timestamps, timestamps + H-1);

	std::cout << "pEst " << pEst << std::endl;
	idxs_x << std::max(float(std::min(arma::min(pEst.row(0)), pS[0] - Vrng[1] * T)), ptrGmap->getXrng()[0]) << std::min(float(std::max(arma::max(pEst.row(0)), pS[0] + Vrng[1] * T)), ptrGmap->getXrng()[1]);
	idxs_y << std::max(float(std::min(arma::min(pEst.row(1)), pS[1] - Vrng[1] * T)), ptrGmap->getYrng()[0]) << std::min(float(std::max(arma::max(pEst.row(1)), pS[1] + Vrng[1] * T)), ptrGmap->getYrng()[1]);
	

	rlen = idxs_y(1) - idxs_y(0);
	clen = idxs_x(1) - idxs_x(0);
	
	std::cout << "idxs_y: " << idxs_y(0) << " / " << idxs_y(1) << std::endl;
	std::cout << "idxs_x: " << idxs_x(0) << " / " << idxs_x(1) << std::endl;

	ptrCCDP = new CCDPmap(ptrGmap, idxs_x, idxs_y, 0.3);
	ptrCCDP->printLocalmap();
	std::cout << "CCDP size : " << (idxs_y(1) - idxs_y(0)) / ptrCCDP->getGrid() << " x " << (idxs_x(1) - idxs_x(0)) / ptrCCDP->getGrid() << std::endl;
	grid = ptrCCDP->getGrid();
	mat_idxs_x = arma::mat(ptrCCDP->getLocalmap());
	mat_idxs_y = arma::mat(ptrCCDP->getLocalmap());
	for (int i = 0; i < mat_idxs_x.n_cols; i++) {mat_idxs_x.col(i).fill(idxs_x(0) + i * grid);}
	for (int i = 0; i < mat_idxs_x.n_rows; i++) {mat_idxs_y.row(i).fill(idxs_y(0) + i * grid);}
	//std::cout << mat_idxs_x << std::endl;
	//std::cout << mat_idxs_y << std::endl;

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
	list_of_mat.clear();

	return Ret;
}



void initCostfunc(float grid) {
	arma::mat local = ptrCCDP->getLocalmap(); 
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
		angle2target = round((arma::atan2(ccdp_ys, ccdp_xs) + pi) / 2.0 / pi*nvis) + 1;
		angle2target.elem(arma::find(angle2target == nvis + 1)).fill(1);
		myPrint(angle2target, false);
		//std::cout << angle2target << std::endl;

		minradius.zeros();
		cvistmp.zeros();
		for (int ii = 0; ii < nvis; ii++) {
			arma::uvec obsidx = arma::find(local == 1 && angle2target == ii);
			if (arma::sum(obsidx) > 0) {
				minradius(ii) = arma::min(dist2target(obsidx));
				cvistmp.elem(arma::find(dist2target >= minradius(ii) && angle2target == ii)).ones();
			}
		}
		Cost_visual[tt + 1] = cvistmp;

		//std::cout << "tt: " << tt << " / Cvis : " << std::endl;
		//myPrint(Cost_visual[tt + 1]);
		//std::cout << cvistmp << std::endl;



		/****************************** [ Ctrack ] *******************************/
		arma::mat afterA(2, local.n_elem);
		afterA.row(0) = arma::vectorise(ccdp_xs, 1);
		afterA.row(1) = arma::vectorise(ccdp_ys, 1);
		std::cout << afterA << std::endl; // 수정해야됨 쮸발!
		arma::mat B;
		arma::mat C(local);

		for (int ii = 0; ii < ndir; ii++) {
			B = (A.slice(ii)*afterA);
			B.each_col() += cc;		
			C.zeros();
			C.elem(arma::find(arma::sum(1 - 0.5 * erfc(B / (-sqrt(2))), 0)>threshold)).ones();
			Cost_track[tt + 1].slice(ii) = arma::reshape(C, local.n_rows, local.n_cols);
			//std::cout << Cost_track[tt + 1].slice(ii) << std::endl;
		}
		//std::cout << Cost_track[tt + 1] << std::endl;


		/****************************** [ Cavoid ] *******************************/
		Cost_avoid[tt + 1] = arma::ceil(arma::conv2(local, mat_conv2, "same"));
		//myPrint(Cost_avoid[tt + 1]);
		//std::cout << Cost_avoid[tt + 1] << std::endl;

		/**************************	*** [ Cconst ] *******************************/
		Cost_const[tt+1] = Cost_track[tt+1].each_slice() + Cost_avoid[tt+1]; // 논문은 +, 맽랩은 or연산
		std::cout << Cost_const[tt+1] << std::endl;
	}

}


void backRecursion(CCDPmap* ptr) {
	arma::mat local = ptr->getLocalmap();
	float grid = ptr->getGrid();


	// -----[Backward recursion] ----- 
	// find the optimal control when lambda is given


	//std::vector<arma::mat> CompMat;

	for (int i = 0; i < H; i++) { 
		Optctrl_row.push_back(arma::cube(local.n_rows, local.n_cols, ndir));
		Optctrl_col.push_back(arma::cube(local.n_rows, local.n_cols, ndir));
		Optctrl_dirs.push_back(arma::cube(local.n_rows, local.n_cols, ndir));
	}
	for (int i = 0; i < H + 1; i++) { Jmin.push_back(arma::cube(local.n_rows, local.n_cols, ndir));}
	Lk = arma::cube(local.n_rows, local.n_cols, ndir);
	Jmin_dir = arma::mat(local).fill(100);
	mindiridx = arma::mat(local).fill(0);
	//for (int i = 0; i < H + 1; i++) { CompMat.push_back(arma::mat()); }

	lagcoeff = 0;
	lagrng = { 0.0, 1.0 };
	lag_feas = 1;

	for (int ilagran = 0; ilagran < 10; ilagran++) {
		for (int i = 0; i < H; i++) {
			Optctrl_row[i].zeros();
			Optctrl_col[i].zeros();
			Optctrl_dirs[i].zeros();
		}
		for (int i = 0; i < H + 1; i++) { Jmin[i].fill(100); }
		for (int i = 0; i < ndir; i++) {
			Jmin[H + 1 - 1].slice(i) = lagcoeff*Cost_const[H + 1 - 1].slice(i) + lag_feas * Cost_visual[H + 1 - 1];
		}

		for (int tt = H - 1; tt >= 0; tt--) {
			for (int i = 0; i < ndir; i++) { Lk.slice(i) = lagcoeff*Cost_const[tt].slice(i) + lag_feas * Cost_visual[tt]; }

			for (int ii = 0; ii < ndir; ii++) {
				Jmin_dir = arma::mat(local).fill(100);
				for (int i = 0; i < nextstate_dir[ii].n_elem; i++) { //assume that nextstate_dir is rowvec or colvec
					for (int j = 0; j < local.n_rows; j++) {
						for (int k = 0; k < local.n_cols; k++) {
							if (Jmin_dir(j, k) > Jmin[tt + 1](j, k, nextstate_dir[ii](i))) {
								Jmin_dir(j, k) = Jmin[tt + 1](j, k, nextstate_dir[ii](i));
								mindiridx(j, k) = nextstate_dir[ii](i);
							}
						}

					}
				}

				for (int jj = 0; jj < nextstate_sidxs[ii].n_rows; jj++) {
					std::vector<int> original_rng_r = { std::max(0,int(0 - nextstate_sidxs[ii](jj,0))), std::min(int(local.n_rows - 1), int(local.n_rows - 1 - nextstate_sidxs[ii](jj,0))) };
					std::vector<int> original_rng_c = { std::max(0,int(0 - nextstate_sidxs[ii](jj,1))), std::min(int(local.n_cols - 1), int(local.n_cols - 1 - nextstate_sidxs[ii](jj,1))) };
					std::vector<int> comp_rng_r = { std::max(0,int(0 + nextstate_sidxs[ii](jj,0))), std::min(int(local.n_rows - 1), int(local.n_rows - 1 + nextstate_sidxs[ii](jj,0))) };
					std::vector<int> comp_rng_c = { std::max(0,int(0 + nextstate_sidxs[ii](jj,1))), std::min(int(local.n_cols - 1), int(local.n_cols - 1 + nextstate_sidxs[ii](jj,1))) };



					arma::mat CompMat(original_rng_r[1] - original_rng_r[0], original_rng_c[1] - original_rng_c[0]);
					for (int i = 0; i < CompMat.n_rows; i++) {
						for (int j = 0; j < CompMat.n_cols; j++) {
							CompMat(i, j) = (Jmin[tt](i + original_rng_r[0], j + original_rng_c[0], ii) > Jmin_dir(i + comp_rng_r[0], j + comp_rng_c[0])) || (Jmin[tt](i + original_rng_r[0], j + original_rng_c[0], ii) == Jmin_dir(i + comp_rng_r[0], j + comp_rng_c[0]) && abs(Optctrl_dirs[tt](i + original_rng_r[0], j + original_rng_c[0], ii)) > abs(mindiridx(i + comp_rng_r[0], j + comp_rng_c[0])));
						}
					}
					for (int i = 0; i < CompMat.n_rows; i++) {
						for (int j = 0; j < CompMat.n_cols; j++) {
							Jmin[tt](i + original_rng_r[0], j + original_rng_c[0], ii) = (1 - CompMat(i, j))*Jmin[tt](i + original_rng_r[0], j + original_rng_c[0], ii) + CompMat(i, j)*Jmin_dir(i + comp_rng_r[0], j + comp_rng_c[0]);
							Optctrl_row[tt](i + original_rng_r[0], j + original_rng_c[0], ii) = (!CompMat(i, j)) * Optctrl_row[tt](i + original_rng_r[0], j + original_rng_c[0], ii) + CompMat(i, j) * nextstate_sidxs[ii](jj, 0);
							Optctrl_col[tt](i + original_rng_r[0], j + original_rng_c[0], ii) = (!CompMat(i, j)) * Optctrl_col[tt](i + original_rng_r[0], j + original_rng_c[0], ii) + CompMat(i, j) * nextstate_sidxs[ii](jj, 1);
							Optctrl_dirs[tt](i + original_rng_r[0], j + original_rng_c[0], ii) = (!CompMat(i, j)) * Optctrl_dirs[tt](i + original_rng_r[0], j + original_rng_c[0], ii) + CompMat(i, j) * mindiridx(i + comp_rng_r[0], j + comp_rng_c[0]);
						}
					}
					//std::cout << CompMat << std::endl;

				}

				//std::cout << nextstate_dir[ii].n_elem << std::endl;
				// for jj 1:size...

				//myPrint(Jmin_dir,false);
				//myPrint(mindiridx, false);
				//system("PAUSE");
				
			}
			Jmin[tt] = Jmin[tt] + Lk;
		}

		currs_idx(0, 0) = (s[1] - idxs_y[0]) / grid;
		currs_idx(1, 0) = (s[0] - idxs_x[0]) / grid;
		currs_idx(2, 0) = int(pS[2] / 2.0 / pi*ndir) % ndir - 1;
		if (currs_idx(2, 0) == -1) currs_idx(2, 0) = 9;
		int Costconst_val = 0;
		optidx.zeros();
		for (int tt = 0; tt < H; tt++) {
			optidx(0, tt) = Optctrl_row[tt](currs_idx(0, tt), currs_idx(1, tt), currs_idx(2, tt));
			optidx(1, tt) = Optctrl_col[tt](currs_idx(0, tt), currs_idx(1, tt), currs_idx(2, tt));
			optidx(2, tt) = Optctrl_dirs[tt](currs_idx(0, tt), currs_idx(1, tt), currs_idx(2, tt));
			currs_idx(0, tt + 1) = currs_idx(0, tt) + optidx(0, tt);
			currs_idx(1, tt + 1) = currs_idx(1, tt) + optidx(1, tt);
			currs_idx(2, tt + 1) = optidx(2, tt);
			Costconst_val = Costconst_val + Cost_const[tt + 1](currs_idx(0, tt + 1), currs_idx(1, tt + 1), currs_idx(2, tt + 1));
		}
		for (int i = 0; i < H + 1; i++) {
			predict_s(0, i) = (currs_idx(1, i) - 1)*grid + idxs_x[0];
			predict_s(1, i) = (currs_idx(0, i) - 1)*grid + idxs_y[0];
			predict_s(2, i) = (currs_idx(2, i) - 1)*2.0*pi / ndir;
		}
		if (Costconst_val <= 0 && ilagran == 0) {
			std::cout << "feasible //  " << ilagran << " times" <<  std::endl;
			break;
		}
		else if (ilagran == 0) {
			lag_feas = 0;
			lagcoeff = 1;
		}

		if (Costconst_val > 0 && ilagran == 1) {
			std::cout << "infeasible //  " << ilagran << " times" << std::endl;
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
				std::cout << ilagran << " times" << std::endl;
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
	

	pS[0] = s[0] + cos(phi)*uv_opt*T;
	pS[1] = s[1] + sin(phi)*uv_opt*T;
	pS[2] = fmod(phi + uw_opt*T, 2.0*pi);
	for (int i = 0; i < 2; i++) { s[i] = pS[i]; }
	phi = pS[2];
	pSlog.push_back(pS);
	pTlog.push_back(pT_est);
	std::cout << std::endl;
	std::cout << "  V: " << uv_opt << "  U: " << uw_opt << std::endl;
	std::cout << "human position: (" << pT_est[0] << ", " << pT_est[1] << ")" << std::endl;
	std::cout << "robot position: (" << pS[0] << ", " << pS[1] << ", " << pS[2]*180.0/2.0/pi << ", num: "<< currs_idx(2, 0) << " " << optidx(2,0) <<  ")"  << std::endl;
	myPrint(Cost_track[1].slice(currs_idx(2, 0)));



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
			if (j == int((round(pT_est[0]) - ptrGmap->getXrng()[0])/grid) && i == int((round(pT_est[1]) - ptrGmap->getYrng()[0]) / grid)) {
				printf(" H ");
			}
			else if (j == int((round(s[0]) - ptrGmap->getXrng()[0]) / grid) && i == int((round(s[1]) - ptrGmap->getYrng()[0]) / grid)) {
				printf(" R ");
			}
			else if (j >= (idxs_x[0] - int(ptrGmap->getXrng()[0]))/grid && j < abs(idxs_x[1] - int(ptrGmap->getXrng()[0])) / grid && i >= (idxs_y[0] - int(ptrGmap->getYrng()[0]))/grid && i < (idxs_y[1] - int(ptrGmap->getYrng()[0])) / grid) {
				if (Cost_track[1].slice(currs_idx(2, 0))(i - int((round(idxs_y[0]) - (ptrGmap->getYrng()[0]))/grid), j - int((round(idxs_x[0]) - (ptrGmap->getXrng()[0]))/grid)) == 0) {
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

