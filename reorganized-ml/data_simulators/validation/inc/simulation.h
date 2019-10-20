//C++ source code for synthetic pulse generation
//by Aaron Sprow <aaron.sprow@uky.edu> June 4, 2018
//Install FFTW libraries and set PATH variable <www.fftw.org>
//g++ -O3 -c simulation.h -std=c++11
//g++ -O3 simulation.o main.cpp -std=c++11 -lfftw (or -lfftw3)
#include <random>
#include <fftw3.h>

using namespace std;
#ifndef PULSE_GEN_INCLUDE
#define PULSE_GEN_INCLUDE

double Box_Muller(default_random_engine &gen, uniform_real_distribution<double> &dist); //generates normally distributed random numbers from uniform distribution
//generates a synthetic pulse shape where a semi-gaussian current pulse is integrated and run through a CR-(RC)^2 shaper
void gen_synth_pulse(short* wf, default_random_engine &gen, 
		uniform_real_distribution<double> &dist); 
//generates a synthetic pulse shape where a semi-gaussian current pulse is integrated and run through a CR-(RC)^2 shaper
void add_pileup(short* wf, default_random_engine &gen, 
		uniform_real_distribution<double> &dist, geometric_distribution<int> &delay); 
void gen_sim_pulse(int amp, short *wf); //generates a simulated pulse shape based on MC output defining energy deposition profile for incident particles

//adds noise to generated pulse as defined by data/power_spectrum.dat
void superimpose_noise(short* wf, 
		default_random_engine &gen, uniform_real_distribution<double> &dist,
		fftw_complex* array, fftw_plan *p); 

#endif
