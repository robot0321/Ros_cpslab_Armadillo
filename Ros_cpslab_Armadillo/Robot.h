#pragma once
#define PI 3.141593

class Robot {
private:
	float Rs, ThetaS; //sensor model
	int Nside;
	float Vmax, Vmin;
	float Umax, Umin;

public:
	float x, y, phi; //q

	Robot();
	Robot(float* q_init, float* sensor_model, int N_side, float* v_rng, float* w_rng);
	~Robot();

	float get_Rs() const;
	float get_ThetaS() const;
	int get_Nside() const;
	float get_Vmax() const;
	float get_Vmin() const;
	float get_Umax() const;
	float get_Umin() const;
};