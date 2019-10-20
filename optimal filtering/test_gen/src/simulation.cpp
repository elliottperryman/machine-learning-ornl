#include "simulation.h"
#include <fftw3.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
using namespace std;

double PI=4.*atan(1.);

double Box_Muller(default_random_engine &gen, uniform_real_distribution<double> &dist) //produces random numbers according to normal distribution
{
	return sqrt(-2.*log(dist(gen)))*cos(2.*PI*dist(gen));
}

void gen_synth_pulse(double amp, int length, short* wf, default_random_engine &gen, 
		uniform_real_distribution<double> &dist)
{
	double cc_slow = 2.5+0.4*Box_Muller(gen,dist); cc_slow=cc_slow/(cc_slow+1); //charge collection slow time constant
	double cc_fast = 1./2.5; //charge collection fast time constant
	double alpha_cr = 1250./(1250.+1.); //fall time of output
	double alpha_rc1 = 1./2.75;
	double alpha_rc2 = 1./2.75;
	double step[2]={0,0},charge[2]={0,0},cur_s[2]={0,0},cur_f[2]={0,0},cr[2]={0,0},rc1[2]={0,0},rc2[2]={0,0};
	int T0=1000;//1000+2.5*Box_Muller();
	for(int i=0; i<length; i++)
	{
		step[i%2]= i>=T0 ? 1. : 0.;
		cur_s[i%2]=cc_slow*(cur_s[(i+1)%2]+step[i%2]-step[(i+1)%2]);
		cur_f[i%2]=cc_fast*(cur_s[i%2]-cur_f[(i+1)%2])+cur_f[(i+1)%2];
		charge[i%2]=charge[(i+1)%2]+amp*cur_f[i%2]*(1./cc_slow-1.);
		cr[i%2]=alpha_cr*(cr[(i+1)%2]+charge[i%2]-charge[(i+1)%2]);
		rc1[i%2]=alpha_rc1*(cr[i%2]-rc1[(i+1)%2])+rc1[(i+1)%2];
		rc2[i%2]=alpha_rc2*(rc1[i%2]-rc2[(i+1)%2])+rc2[(i+1)%2];
		wf[i]=(short)rc2[i%2];
	}
	return;
}

void add_pileup(double amp, int length, short* wf, default_random_engine &gen, 
		uniform_real_distribution<double> &dist)
{
	double cc_slow = 2.5+0.4*Box_Muller(gen,dist); cc_slow=cc_slow/(cc_slow+1); //charge collection slow time constant
	double cc_fast = 1./2.5; //charge collection fast time constant
	double alpha_cr = 1250./(1250.+1.); //fall time of output
	double alpha_rc1 = 1./2.75;
	double alpha_rc2 = 1./2.75;
	double step[2]={0,0},charge[2]={0,0},cur_s[2]={0,0},cur_f[2]={0,0},cr[2]={0,0},rc1[2]={0,0},rc2[2]={0,0};
	
	int T0=1000 + 100 + 2.5*Box_Muller(gen, dist);
	for(int i=0; i<length; i++)
	{
		step[i%2]= i>=T0 ? 1. : 0.;
		cur_s[i%2]=cc_slow*(cur_s[(i+1)%2]+step[i%2]-step[(i+1)%2]);
		cur_f[i%2]=cc_fast*(cur_s[i%2]-cur_f[(i+1)%2])+cur_f[(i+1)%2];
		charge[i%2]=charge[(i+1)%2]+amp*cur_f[i%2]*(1./cc_slow-1.);
		cr[i%2]=alpha_cr*(cr[(i+1)%2]+charge[i%2]-charge[(i+1)%2]);
		rc1[i%2]=alpha_rc1*(cr[i%2]-rc1[(i+1)%2])+rc1[(i+1)%2];
		rc2[i%2]=alpha_rc2*(rc1[i%2]-rc2[(i+1)%2])+rc2[(i+1)%2];
		wf[i]+=(short)rc2[i%2];
	}
	return;
}



void gen_sim_pulse(int amp, int length, short *wf)
{
	double X0=7485000., Y0=7485000., Z0=0.; //location of initial carrier production in silicon, in nm
	double DET_TEMP=110; //temperature in K
	double V_ELEC, V_HOLE; //drift velocities for charge carriers
	double SPEED_OF_LIGHT=299792458.; //speed of light in m/s
	double V_INCIDENT; //velocity of incident particle, calculated each step using c*sqrt(1-1/(1+KE/mc^2)^2), where KE = avg(E_curr, E_prev)
	double EGEN_PAIR=0.00381+DET_TEMP*(0.00381-0.0036);
	return;
}

void superimpose_noise(int length, short* wf, default_random_engine &gen, uniform_real_distribution<double> &dist)
{
	ifstream fin("config/power_spectrum.dat");
	double *powerspectrum = new double[32766];
	for(int i=0; i<32766; i++)
		fin>>powerspectrum[i];
	fftw_complex *array = new fftw_complex[65533];
	fftw_plan p;
	double phase;
	array[0][0]=-114.962;
	array[0][1]=0.;
	for(int i=1; i<32767; i++)
	{
		phase = 2.*PI*dist(gen);
		array[65532-i+1][0]=array[i][0]=powerspectrum[i-1]*cos(phase);
		array[65532-i+1][1]=-1*(array[i][1]=powerspectrum[i-1]*sin(phase));
	}
	p = fftw_plan_dft_1d(65533,array,array,FFTW_BACKWARD,FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);
	for(int i=0; i<length; i++)
		wf[i]+=array[i][0];
	return;
}
