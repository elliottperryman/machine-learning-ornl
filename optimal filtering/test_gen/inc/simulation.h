//C++ source code for synthetic pulse generation
//by Aaron Sprow <aaron.sprow@uky.edu> June 4, 2018
//Install FFTW libraries and set PATH variable <www.fftw.org>
//g++ -O3 -c simulation.h -std=c++11
//g++ -O3 simulation.o main.cpp -std=c++11 -lfftw (or -lfftw3)
#include <random>
using namespace std;
#ifndef PULSE_GEN_INCLUDE
#define PULSE_GEN_INCLUDE

double Box_Muller(default_random_engine &gen, uniform_real_distribution<double> &dist); //generates normally distributed random numbers from uniform distribution
void gen_synth_pulse(double amp, int length, short* wf, default_random_engine &gen, uniform_real_distribution<double> &dist); //generates a synthetic pulse shape where a semi-gaussian current pulse is integrated and run through a CR-(RC)^2 shaper
void add_pileup(double amp, int length, short* wf, default_random_engine &gen, uniform_real_distribution<double> &dist); //generates a synthetic pulse shape where a semi-gaussian current pulse is integrated and run through a CR-(RC)^2 shaper
void gen_sim_pulse(int amp, int length, short *wf); //generates a simulated pulse shape based on MC output defining energy deposition profile for incident particles
void superimpose_noise(int length, short* wf, default_random_engine &gen, uniform_real_distribution<double> &dist); //adds noise to generated pulse as defined by data/power_spectrum.dat

#endif
