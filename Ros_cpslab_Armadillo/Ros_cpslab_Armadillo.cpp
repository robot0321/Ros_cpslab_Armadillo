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


void funf(int a);
double normCDF(double value);
arma::mat unique_rows(arma::mat A);
void setCost_visual(int y, int x);
void setCost_avoid(int y, int x);
void setCost_track(int y, int x, int num);
void setCost_const(int y, int x, int num);
void initCostfunc(int y, int x, int num, float grid);

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
	double da = 0.0;
	double db = 0.5;
	double dc = 1.0;

	pseq << 0 << -1 << -1 << -2 << -3 << -4 << -5 << -6 << -7 << -7 << -7 << -7 << -7 << -7 << -7 << -6 << -5 << -4 << -3 << -2 << -1 << 0 << arma::endr
		 << 0 << 0  << 1  << 1  << 1  <<  1 << 1  <<  1 <<  1 <<  2 <<  3 <<  4 <<  5 <<  6 <<  7 <<  7 <<  7 <<  7 <<  7 <<  7 <<  7 << 7 << arma::endr;
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
	std::cout << A << std::endl;

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
			shiftdir_buffer(j) = (int(shiftdir(j)) + i + ndir) % ndir;
			if (shiftdir_buffer(j) == 0) shiftdir_buffer(j) = ndir;
		}
		nextstate_dir.push_back(shiftdir_buffer);
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
	idxs_x = { std::max(std::min(arma::min(pEst.row(0)), pS[0] - Vrng[1] * T*H), ptrGmap->getXrng().at(0)), std::min(std::max(arma::max(pEst.row(0)), pS[0] + Vrng[1] * T*H), ptrGmap->getXrng().at(1)) };
	idxs_y = { std::max(std::min(arma::min(pEst.row(1)), pS[1] - Vrng[1] * T*H), ptrGmap->getYrng().at(0)), std::min(std::max(arma::max(pEst.row(1)), pS[1] + Vrng[1] * T*H), ptrGmap->getYrng().at(1)) };

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

		for (int i = 0; i < dist2target.n_rows; i++) {
			for (int j = 0; j < dist2target.n_cols; j++) {
				angle2target(i, j) = round((std::atan2(pEst(1, tt) - (ptrCCDP->getGrid()*i + idxs_y[0]), pEst(0, tt) - (ptrCCDP->getGrid()*j + idxs_x[0]))+pi)/2/pi*nvis)+1;
				if (angle2target(i, j) == nvis + 1) angle2target(i,j) = 1;
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
					cvistmp(j) =  (dist2target(j) >= minradius(ii)) & (angle2target(j)==ii) || int(cvistmp(j));
				}
			}
			//std::cout << std::endl;
		}
		delete obsidx;
		Cost_visual[tt+1] = cvistmp;

		//myPrint(Cost_visual[tt + 1]);
		//std::cout << "tt: " << tt << " / Cvis : " << std::endl;
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
					Cost_const[tt + 1](i, j, k) = int(Cost_track[tt + 1](i,j,k)) || int(Cost_avoid[tt + 1](i,j));
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

	// -----[Backward recursion] ----- 
	// find the optimal control when lambda is given



	for (int ilagran = 0; ilagran < 100; ilagran++) {
		//initialize the optimal control : 'col' : current column + col,
		// 'row' : current row + 'row', dirs : replace current dirs with dirs
		Optctrl(1:H) = struct('row', zeros([size(idxs_x), ndir]), 'col', zeros([size(idxs_x), ndir]), 'dirs', 0 * ones([size(idxs_x), ndir]));
		Jmin(1:H + 1) = struct('Jmin', 100 * ones([size(idxs_x), ndir]));
		Jmin(H + 1).Jmin = bsxfun(@plus, lagcoeff*ccdp.cconst(H + 1).val, lag_feas*ccdp.cvis(:, : , H + 1)); %Value: Cvis + lambda * Ctrack
			for tt = H : -1 : 1
				% lagrangian - onestep
				Lk = bsxfun(@plus, lagcoeff*ccdp.cconst(tt).val, lag_feas*ccdp.cvis(:, : , tt)); %Value: Cvis + lambda * Ctrack
				for ii = 1 : ndir
					[Jmin_dir, mindiridx] = min(Jmin(tt + 1).Jmin(:, : , ccdp.nextstate(ii).dir), [], 3);
		mindiridx = ccdp.nextstate(ii).dir(mindiridx);



		% find the optimal control - onestep
			for jj = 1:size(ccdp.nextstate(ii).sidxs, 1)
				original_rng_r = max(1, 1 - ccdp.nextstate(ii).sidxs(jj, 1)) : min(rlen, rlen - ccdp.nextstate(ii).sidxs(jj, 1));
		original_rng_c = max(1, 1 - ccdp.nextstate(ii).sidxs(jj, 2)) :min(clen, clen - ccdp.nextstate(ii).sidxs(jj, 2));
		comp_rng_r = max(1, 1 + ccdp.nextstate(ii).sidxs(jj, 1)) :min(rlen, rlen + ccdp.nextstate(ii).sidxs(jj, 1));
		comp_rng_c = max(1, 1 + ccdp.nextstate(ii).sidxs(jj, 2)) :min(clen, clen + ccdp.nextstate(ii).sidxs(jj, 2));

		% compare two matrix
			CompMat = (Jmin(tt).Jmin(original_rng_r, original_rng_c, ii) > Jmin_dir(comp_rng_r, comp_rng_c)) | ...
			(Jmin(tt).Jmin(original_rng_r, original_rng_c, ii) == Jmin_dir(comp_rng_r, comp_rng_c) & abs(Optctrl(tt).dirs(original_rng_r, original_rng_c, ii)) > abs(mindiridx(comp_rng_r, comp_rng_c)));
		% update Jmin
			Jmin(tt).Jmin(original_rng_r, original_rng_c, ii) = (1 - CompMat).*Jmin(tt).Jmin(original_rng_r, original_rng_c, ii) + CompMat.*Jmin_dir(comp_rng_r, comp_rng_c);

		% update optimal control
			Optctrl(tt).row(original_rng_r, original_rng_c, ii) = (~CompMat).*Optctrl(tt).row(original_rng_r, original_rng_c, ii) + CompMat.*ccdp.nextstate(ii).sidxs(jj, 1);
		Optctrl(tt).col(original_rng_r, original_rng_c, ii) = (~CompMat).*Optctrl(tt).col(original_rng_r, original_rng_c, ii) + CompMat.*ccdp.nextstate(ii).sidxs(jj, 2);
		Optctrl(tt).dirs(original_rng_r, original_rng_c, ii) = (~CompMat).*Optctrl(tt).dirs(original_rng_r, original_rng_c, ii) + CompMat.*mindiridx(comp_rng_r, comp_rng_c);

		end
			end
			Jmin(tt).Jmin = Jmin(tt).Jmin + Lk;
		end

			% --[Find the optimal control] -- %
			currs_idx = [round([s(2) - ccdp.ys(1), s(1) - ccdp.xs(1)] / ccdp.glen)'+1; mod(round(pS(3)/2/pi*ndir),ndir)+1];
			%if (currs_idx(3) == 0)
			% currs_idx(3) = ndir;
		%end
			const_val = 0; % number of violations of constraints
			optidx = zeros(3, H);
		for tt = 1:H
			optidx(:, tt) = [Optctrl(tt).row(currs_idx(1, tt), currs_idx(2, tt), currs_idx(3, tt)), Optctrl(tt).col(currs_idx(1, tt), currs_idx(2, tt), currs_idx(3, tt)), Optctrl(tt).dirs(currs_idx(1, tt), currs_idx(2, tt), currs_idx(3, tt))]';
			currs_idx(1:2, tt + 1) = currs_idx(1:2, tt) + optidx(1:2, tt);
		currs_idx(3, tt + 1) = optidx(3, tt);
		const_val = const_val + ccdp.cconst(tt + 1).val(currs_idx(1, tt + 1), currs_idx(2, tt + 1), currs_idx(3, tt + 1));
		end
			predict_s = [(currs_idx(2, :) - 1)*ccdp.glen + ccdp.xs(1); (currs_idx(1, :) - 1)*ccdp.glen + ccdp.ys(1)];
		predict_s(3, :) = (currs_idx(3, :) - 1) * 2 * pi / ndir;

		% though we doesn't consider constraints, the optimal solution
			% satisfies constraints
			if (const_val <= 0 && ilagran == 1)
				fprintf('feasible\n ');
		break;
		elseif(ilagran == 1)
			lag_feas = 0; lagcoeff = 1;

		end
			% check feasibility
			if (ilagran == 2 && const_val > 0)
				fprintf('infeasible\n');
		break;
		elseif(ilagran == 2)
			lagcoeff = lagrng(2);
		lag_feas = 1;
		end
			if (ilagran > 2)
				if (const_val > 0) % violate constraints
					lagrng(1) = lagcoeff;
				else
					lagrng(2) = lagcoeff;
		end
			if (lagrng(2) - lagrng(1) < 1e-2)
				break;  % stop iterations
			else
				lagcoeff = mean(lagrng);
		end
			end
	}





}

