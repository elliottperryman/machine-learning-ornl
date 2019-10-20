#include "simulation.h"
#include <iostream>
#include <time.h>
#include <stdio.h>
#include <chrono>
#include <fstream>

#define length 3500
#define batchSize 100

using namespace std;

int main(int argc, char** argv)
{
	if (argc!=2) {
		perror("usage: ./pulse_generation output_file");
		exit(-1);
	}
	auto start = chrono::steady_clock::now();	
	time_t timer;
	int SEED=time(&timer); //set a value rather than using the time if you want a manual seed for reproducing results
	default_random_engine generator;
	generator.seed(SEED);
	uniform_real_distribution<double> distribution(0.0,1.0);
	short test[length*18];

	//ofstream fout(argv[1], ios::out);	
	ofstream cout_data(argv[1], ios::out);	

	fftw_complex *array = new fftw_complex[65533];
	fftw_plan p;

	fftw_import_wisdom_from_filename("wisdom/wisdom.data");	
	p = fftw_plan_dft_1d(65533,array,array,FFTW_BACKWARD,FFTW_ESTIMATE);

	double amp;
	double osc_amp, osc_phase, osc_freq;
	int i,j;
	

	#pragma unroll
	for(j=0; j<batchSize; j++)
	{
		osc_amp = distribution(generator)*30.0 - 15;
		osc_freq = distribution(generator)*0.0014+0.002;
		osc_phase = distribution(generator)*2*3.1425;

		//amp = distribution(generator)*2000.+500;		// energy	
		gen_synth_pulse(test,generator,distribution);
		superimpose_noise(test,generator,distribution, array, &p);
		#pragma unroll
		for (int k=0;k<18;k++) {	
			#pragma unroll
			for(i=0; i<length; i++)
				test[i+k*length] += osc_amp*sin(osc_freq*i+osc_phase);
			#pragma unroll
			for(i=0; i<length; i++)
				cout_data<<test[i+k*length]<<" ";
			cout_data<<"\n";
		}	
	}
	#pragma unroll
	for(j=0; j<batchSize; j++)
	{
		osc_amp = distribution(generator)*30.0 - 15;
		osc_freq = distribution(generator)*0.0014+0.002;
		osc_phase = distribution(generator)*2*3.1425;

		add_pileup(test,generator,distribution);
		superimpose_noise(test,generator,distribution, array, &p);
		#pragma unroll
		for (int k=0;k<18;k++) {	
			#pragma unroll
			for(i=0; i<length; i++)
				test[i+k*length] += osc_amp*sin(osc_freq*i+osc_phase);
			#pragma unroll
			for(i=0; i<length; i++)
				cout_data<<test[i+k*length]<<" ";
			cout_data<<"\n";
		}
	}

	auto end = chrono::steady_clock::now();	
	fprintf(stderr, "Run Time: %ld\n", chrono::duration_cast<chrono::milliseconds>(end-start).count());	

	return 0;
}
