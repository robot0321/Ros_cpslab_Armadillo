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


float Rs = 5; 
float ThetaS = 57.0 / 180.0 * pi;
float sensor_model[2] = { Rs, ThetaS };
int Nside = 4;
float VMAX_const = 4;
float VMIN = -4;
arma::fmat Urng(1, 2); 
arma::fmat Vrng(1, 2); 
const int H = 3; // horizon
const int T = 1;
const float threshold = 0.5;
int ndir = 10;
int nvis = 20;
float pT_est[2];
float s[2];
float phi = 2 * pi / ndir;
float pS[3];
float Rfree = 1;
float lagcoeff = 0;
std::vector<float> lagrng = { 0.0, 1.0 };
float lag_feas = 1;
std::vector<float> Optctrl;
arma::mat mat_conv2 = arma::mat().ones(int(Rfree * 2 + 1), int(Rfree * 2 + 1));


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
std::vector<float*> pSlog;
std::vector<float*> pTlog;

void funf(int a);
double normCDF(double value);
arma::mat unique_rows(arma::mat A);
void initCostfunc(int y, int x, int num, float grid);
void backRecursion(CCDPmap* ptr);

void myPrint(const arma::mat M, bool obj=true) {
	for (int i = M.n_rows-1; i >= 0; i--) {
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
	int ttmp = 1;
	Urng(0) = -pi / 2;
	Urng(1) = pi / 2;
	Vrng(0) = VMIN;
	Vrng(1) = VMAX_const;

	s[0] = pseq(0, ttmp) - 1;
	s[1] = pseq(1, ttmp) - 1;
	phi = 2.0*pi/ndir;
	pS[0] = s[0];
	pS[1] = s[1];
	pS[2] = phi;

	for (int i = 0; i < ndir; i++) { s_dir(i) = 2 * PI*i / ndir; }

	for (int i = 0; i < Nside; i++) {//setting phi_a
		if (i == 0) { phi_a(0) = (ThetaS / 2 + pi / 2); cc(i) = 0; }
		else if (i == 1) { phi_a(1) = (-ThetaS / 2 - pi / 2); cc(i) = 0; }
		else { phi_a(i) = ThetaS*(2.0 * (i+1) - 3 - Nside) / (2.0 * (Nside - 2)); cc(i) = Rs * cos(ThetaS / 2.0 / (Nside - 2)); }
	}


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
	backRecursion(ptrCCDP);






	delete ptrCCDP;

}

double normCDF(double value)
{
	return 0.5 * erfc(-value/sqrt(2));
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

		for (int i = 0; i < dist2target.n_rows; i++) {
			for (int j = 0; j < dist2target.n_cols; j++) {
				angle2target(i, j) = round((std::atan2(pEst(1, tt) - (ptrCCDP->getGrid()*i + idxs_y[0]), pEst(0, tt) - (ptrCCDP->getGrid()*j + idxs_x[0])) + pi) / 2 / pi*nvis) + 1;
				if (angle2target(i, j) == nvis + 1) angle2target(i, j) = 1;
			}
		}
		//myPrint(angle2target, false);
		//std::cout << angle2target << std::endl;

		int* obsidx = new int[local.n_cols * local.n_rows];
		for (int ii = 0; ii < nvis; ii++) {
			float dist_buff = pow(ptrGmap->getXrng().at(1) - ptrGmap->getXrng().at(0), 2) + pow(ptrGmap->getYrng().at(1) - ptrGmap->getYrng().at(0), 2);
			int sum_obsidx = 0;
			for (int j = 0; j < local.n_rows*local.n_cols; j++) {
				obsidx[j] = (local(j) == 1 & angle2target(j) == ii);
				tmp(j) = obsidx[j];
				sum_obsidx += obsidx[j];
				if (obsidx[j] == 1) {
					dist_buff = std::fmin(dist2target(j), dist_buff);
				}
			}
			//std::cout << std::endl;
			//std::cout << tmp << std::endl;

			if (sum_obsidx > 0) {
				minradius(ii) = dist_buff;
				for (int j = 0; j < local.n_rows*local.n_cols; j++) {
					cvistmp(j) = (dist2target(j) >= minradius(ii)) & (angle2target(j) == ii) || int(cvistmp(j));
				}
			}
			//std::cout << std::endl;
		}
		delete obsidx;
		Cost_visual[tt + 1] = cvistmp;

		//std::cout << "tt: " << tt << " / Cvis : " << std::endl;
		//myPrint(Cost_visual[tt + 1]);
		//std::cout << cvistmp << std::endl;



		/****************************** [ Ctrack ] *******************************/
		arma::mat aass(2, local.n_cols*local.n_rows);
		arma::mat track_tmp(local);
		track_tmp.ones();

		for (int ii = 0; ii < ndir; ii++) {
			for (int i = 0; i < local.n_rows; i++) {
				for (int j = 0; j < local.n_cols; j++) {
					aass(0, local.n_cols*i + j) = (ptrCCDP->getGrid()*i + idxs_x[0]) - pEst(0, tt);
					aass(1, local.n_cols*i + j) = (ptrCCDP->getGrid()*j + idxs_y[0]) - pEst(1, tt);
				}
			}
			//std::cout << "aass" << std::endl;
			//std::cout << aass << std::endl;
			//std::cout << "cc" << std::endl;
			//std::cout << cc << std::endl;

			arma::mat bbss = A.slice(ii) * aass;
			//std::cout << "A.slice * aass" << std::endl;
			//std::cout << bbss << std::endl;

			for (int i = 0; i < bbss.n_cols; i++) {
				bbss.col(i) += cc;
			}

			//std::cout << "+cc" << std::endl;
			//std::cout << bbss << std::endl;
			for (int i = 0; i < bbss.n_rows; i++) {
				for (int j = 0; j < bbss.n_cols; j++) {
					bbss(i, j) = 1 - normCDF(bbss(i, j));
				}
			}
			arma::mat b(local.n_rows, local.n_cols);

			//std::cout << "normcdf 1" << std::endl;
			//std::cout << normCDF(1) << std::endl;

			//std::cout << "cdf" << std::endl;
			//std::cout << bbss << std::endl;

			bbss = arma::sum(bbss, 0);
			//std::cout << "sum" << std::endl;
			//std::cout << bbss << std::endl;
			for (int i = 0; i < bbss.n_rows; i++) {
				for (int j = 0; j < bbss.n_cols; j++) {
					b(bbss.n_cols*i + j) = (bbss(i, j) > threshold);
				}
			}
			/*don't need cuz Cost_const
			for (int i = 0; i < b.n_rows; i++) {
				for (int j = 0; j < b.n_cols; j++) {
					b(i, j) = int(b(i, j)) || int(ptrCCDP->getLocalmap()(i, j));
				}
			}
			*/

			Cost_track[tt + 1].slice(ii) = b;
			//myPrint(b);

			/* don't need cuz Cost_const
			for (int i = 0; i < b.n_rows; i++) {
				for (int j = 0; j < b.n_cols; j++) {
					track_tmp(i,j) = int(track_tmp(i,j)) & int(b(i, j));
				}
			}
			*/

		}
		//myPrint(track_tmp);		


		/****************************** [ Cavoid ] *******************************/
		Cost_avoid[tt + 1] = arma::trunc(arma::conv2(local, mat_conv2, "same"));
		//myPrint(Cost_avoid[tt + 1]);


		/****************************** [ Cconst ] *******************************/
		for (int k = 0; k < ndir; k++) {
			for (int i = 0; i < local.n_rows; i++) {
				for (int j = 0; j < local.n_cols; j++) {
					Cost_const[tt + 1](i, j, k) = int(Cost_track[tt + 1](i, j, k)) || int(Cost_avoid[tt + 1](i, j));
				}
			}
		}

		/* Cost_Const print
		for (int i = 0; i < ndir; i++) {
			//myPrint(Cost_const[tt + 1].slice(i));
			for (int j = 0; j < local.n_rows; j++) {
				for (int k = 0; k < local.n_cols; k++) {
					track_tmp(j,k) = track_tmp(j,k) && Cost_const[tt + 1].slice(i)(j,k);
				}
			}

		}
		myPrint(track_tmp);
		system("PAUSE");
		*/
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


				}

				//std::cout << nextstate_dir[ii].n_elem << std::endl;
				// for jj 1:size...

				//myPrint(Jmin_dir,false);
				//myPrint(mindiridx, false);
				//system("PAUSE");
			}
			Jmin[tt] = Jmin[tt] + Lk;
		}

		currs_idx(0, 0) = round((s[1] - idxs_y[0]) / grid);
		currs_idx(1, 0) = round((s[0] - idxs_x[0]) / grid);
		currs_idx(2, 0) = (int(round(pS[3] / 2.0 / pi*ndir)) % ndir) - 1;
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
			std::cout << "feasible" << std::endl;
			break;
		}
		else if (ilagran == 0) {
			lag_feas = 0;
			lagcoeff = 1;
		}

		if (Costconst_val > 0 && ilagran == 1) {
			std::cout << "infeasible" << std::endl;
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
	pS[2] = phi + uw_opt*T;
	for (int i = 0; i < 2; i++) { s[i] = pS[i]; }
	phi = pS[2];
	pSlog.push_back(pS);
	pTlog.push_back(pT_est);
}