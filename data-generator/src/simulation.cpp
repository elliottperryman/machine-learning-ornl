#include "simulation.h"
#include <fftw3.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>

#include "powerSpectrum.h"
#define length 3500

using namespace std;

double PI=4.*atan(1.);

//produces random numbers according to normal distribution
double Box_Muller(default_random_engine &gen, uniform_real_distribution<double> &dist) 
{
	return sqrt(-2.*log(dist(gen)))*cos(2.*PI*dist(gen));
}

void gen_synth_pulse(short* wf, default_random_engine &gen, 
		uniform_real_distribution<double> &dist)
{
	#pragma unroll	
	for (int k=0;k<18;k++) {	
		double cc_slow = 2.5+0.4*Box_Muller(gen,dist); cc_slow=cc_slow/(cc_slow+1); //charge collection slow time constant
		double cc_fast = 1./2.5; //charge collection fast time constant
		double alpha_cr = 1250./(1250.+1.); //fall time of output
		double alpha_rc1 = 1./2.75;
		double alpha_rc2 = 1./2.75;
		double step[2]={0,0},charge[2]={0,0},cur_s[2]={0,0},cur_f[2]={0,0},cr[2]={0,0},rc1[2]={0,0},rc2[2]={0,0};
		int T0 = 900+dist(gen)*200;
		double amp = dist(gen)*2300.+200.;
		#pragma unroll	
		for(int i=0; i<length; i++)
		{
			step[i&0x1]= i>=T0 ? 1. : 0.;
			cur_s[i&0x1]=cc_slow*(cur_s[(i&0x1)^0x1]+step[i&0x1]-step[(i&0x1)^0x1]);
			cur_f[i&0x1]=cc_fast*(cur_s[i&0x1]-cur_f[(i&0x1)^0x1])+cur_f[(i&0x1)^0x1];
			charge[i&0x1]=charge[(i&0x1)^0x1]+amp*cur_f[i&0x1]*(1./cc_slow-1.);
			cr[i&0x1]=alpha_cr*(cr[(i&0x1)^0x1]+charge[i&0x1]-charge[(i&0x1)^0x1]);
			rc1[i&0x1]=alpha_rc1*(cr[i&0x1]-rc1[(i&0x1)^0x1])+rc1[(i&0x1)^0x1];
			rc2[i&0x1]=alpha_rc2*(rc1[i&0x1]-rc2[(i&0x1)^0x1])+rc2[(i&0x1)^0x1];
			wf[i+k*length]=(short)rc2[i&0x1];
		}
	}	
	return;
}

void add_pileup(short* wf, default_random_engine &gen, 
		uniform_real_distribution<double> &dist)
{	
	#pragma unroll	
	for (int k=0;k<18;k++) {	
		// generate first bunch	
		double cc_slow = 2.5+0.4*Box_Muller(gen,dist); cc_slow=cc_slow/(cc_slow+1); 
		double cc_fast = 1./2.5; //charge collection fast time constant
		double alpha_cr = 1250./(1250.+1.); //fall time of output
		double alpha_rc1 = 1./2.75;
		double alpha_rc2 = 1./2.75;
		double step[2]={0,0},charge[2]={0,0},cur_s[2]={0,0},cur_f[2]={0,0};
		double cr[2]={0,0},rc1[2]={0,0},rc2[2]={0,0};

		int T0=900+dist(gen)*200;
		double energy = dist(gen)*2300+200;	
		double perc1 = dist(gen)*0.8 + 0.1;	
		double perc2 = 1.0-perc1;	
		double amp = perc1*energy;			
		#pragma unroll	
		for(int i=0; i<length; i++)
		{
			step[i&0x1]= i>=T0 ? 1. : 0.;
			cur_s[i&0x1]=cc_slow*(cur_s[(i&0x1)^0x1]+step[i&0x1]-step[(i&0x1)^0x1]);
			cur_f[i&0x1]=cc_fast*(cur_s[i&0x1]-cur_f[(i&0x1)^0x1])+cur_f[(i&0x1)^0x1];
			charge[i&0x1]=charge[(i&0x1)^0x1]+amp*cur_f[i&0x1]*(1./cc_slow-1.);
			cr[i&0x1]=alpha_cr*(cr[(i&0x1)^0x1]+charge[i&0x1]-charge[(i&0x1)^0x1]);
			rc1[i&0x1]=alpha_rc1*(cr[i&0x1]-rc1[(i&0x1)^0x1])+rc1[(i&0x1)^0x1];
			rc2[i&0x1]=alpha_rc2*(rc1[i&0x1]-rc2[(i&0x1)^0x1])+rc2[(i&0x1)^0x1];
			wf[i+k*length]=(short)rc2[i&0x1];
		}
		
		// generate pilup on top of that	
		step[0]=0;step[1]=0;
		charge[0]=0;charge[1]=0;
		cur_s[0]=0;cur_s[1]=0;
		cur_f[0]=0;cur_f[1]=0;
		cr[0]=0;cr[1]=0;
		rc1[0]=0;rc1[1]=0;	
		rc2[0]=0;rc2[1]=0;	

		amp = perc2*energy;			
		T0 += dist(gen)*38+2; // delay
		#pragma unroll	
		for(int i=0; i<length; i++)
		{
			step[i&0x1]= i>=T0 ? 1. : 0.;
			cur_s[i&0x1]=cc_slow*(cur_s[(i&0x1)^0x1]+step[i&0x1]-step[(i&0x1)^0x1]);
			cur_f[i&0x1]=cc_fast*(cur_s[i&0x1]-cur_f[(i&0x1)^0x1])+cur_f[(i&0x1)^0x1];
			charge[i&0x1]=charge[(i&0x1)^0x1]+amp*cur_f[i&0x1]*(1./cc_slow-1.);
			cr[i&0x1]=alpha_cr*(cr[(i&0x1)^0x1]+charge[i&0x1]-charge[(i&0x1)^0x1]);
			rc1[i&0x1]=alpha_rc1*(cr[i&0x1]-rc1[(i&0x1)^0x1])+rc1[(i&0x1)^0x1];
			rc2[i&0x1]=alpha_rc2*(rc1[i&0x1]-rc2[(i&0x1)^0x1])+rc2[(i&0x1)^0x1];
			wf[i+k*length]+=(short)rc2[i&0x1];
		}
	
	}	

}


void superimpose_noise(short* wf, default_random_engine &gen, 
		uniform_real_distribution<double> &dist, fftw_complex* array,
		fftw_plan *p)
{
	array[0][0]=-114.962;
	array[0][1]=0.;

	double phase;

	#pragma unroll	
	for(int i=1; i<32767; i++)
	{
		phase = 2.*PI*dist(gen);
		array[65532-i+1][0]=array[i][0]=powerspectrum[i-1]*cos(phase);
		array[65532-i+1][1]=-1*(array[i][1]=powerspectrum[i-1]*sin(phase));
	}

	fftw_execute(*p);
	#pragma unroll	
	for(int i=0; i<length*18; i++)
		wf[i]+=2.5*array[i][0];
	return;
}
