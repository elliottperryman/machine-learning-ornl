#include "simulation.h"
#include <iostream>
#include <time.h>
#include <stdio.h>
#include <chrono>

#define length 3500
#define batchSize 100

using namespace std;

int main(int argc, char** argv)
{
	auto start = chrono::steady_clock::now();	
	time_t timer;
	int SEED=0; //set a value rather than using the time if you want a manual seed for reproducing results
	default_random_engine generator;
	generator.seed(SEED);
	uniform_real_distribution<double> distribution(0.0,1.0);
	short test[length*18];
	
	fftw_complex *array = new fftw_complex[65533];
	fftw_plan p;
	//double phase;
	//array[0][0]=-114.962;
	//array[0][1]=0.;
	p = fftw_plan_dft_1d(65533,array,array,FFTW_BACKWARD,FFTW_PATIENT);
	double amp;
	double perc;
	double amp1;
	double amp2;
	int i,j;

	#pragma unroll
	for(j=0; j<batchSize; j++)
	{
		amp = distribution(generator)*2000.+500;	
		gen_synth_pulse(amp, test,generator,distribution);
		superimpose_noise(test,generator,distribution, array, &p);
		for (int k=0;k<18;k++) {	
			for(i=0; i<length; i++)
				cout<<test[i]<<" ";
			cout<<"\n";
		}	
	}
#pragma unroll
	for(j=0; j<batchSize; j++)
	{
		amp = distribution(generator)*2000.+500;	
		perc = distribution(generator);
		amp1 = amp*perc;
		amp2 = amp*(1.-perc);

		gen_synth_pulse(amp1, test,generator,distribution);
		add_pileup(amp2, test,generator,distribution);
		superimpose_noise(test,generator,distribution, array, &p);
		for (int k=0;k<18;k++) {	
			for(i=0; i<length; i++)
				cout<<test[i]<<" ";
			cout<<"\n";
		}
	}

	auto end = chrono::steady_clock::now();	
	fprintf(stderr, "Run Time: %ld\n", chrono::duration_cast<chrono::milliseconds>(end-start).count());	
	//fftw_destroy_plan(p);
	return 0;
}
