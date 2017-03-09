#include "stdafx.h"
#include "Robot.h"

Robot::Robot(){
	x=0; y=0; phi=0; //q
	Rs=7, ThetaS=57/180*PI; //sensor model
	Nside=4;
	Vmax=4; Vmin=-4; 
	Umax=PI/2.0; Umin=-PI/2.0;
}

Robot::Robot(float* q_init, float* sensor_model, int N_side, float* v_rng, float* w_rng){
	x=q_init[0]; y=q_init[1]; phi=q_init[2]; //q
	Rs=sensor_model[0]; ThetaS=sensor_model[1]; //sensor model
	Nside = N_side;
	Vmax=v_rng[0]; Vmin=v_rng[1]; 
	Umax=w_rng[0]; Umin=w_rng[1];
}

Robot::~Robot(){

}

float Robot::get_Rs() const{
	return Rs;
}
float Robot::get_ThetaS() const{
	return ThetaS;
}

int Robot::get_Nside() const{
	return Nside;
}

float Robot::get_Vmax() const{
	return Vmax;
}

float Robot::get_Vmin() const{
	return Vmin;
}

float Robot::get_Umax() const{
	return Umax;
}

float Robot::get_Umin() const{
	return Umin;
}

