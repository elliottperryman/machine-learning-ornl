#include "simulation.h"
#include <iostream>
#include <time.h>
#include <stdio.h>
#include <chrono>

using namespace std;

int main(int argc, char** argv)
{
	auto start = chrono::steady_clock::now();	
	time_t timer;
	int SEED=0; //set a value rather than using the time if you want a manual seed for reproducing results
	default_random_engine generator;
	generator.seed(SEED);
	uniform_real_distribution<double> distribution(0.0,1.0);
	short test[3500];
	for(int j=0; j<1000; j++)
	{
		double amp = distribution(generator)*2000.+500;	
		gen_synth_pulse(amp, 3500,test,generator,distribution);
		superimpose_noise(3500,test,generator,distribution);
		for(int i=0; i<3500; i++)
			cout<<test[i]<<" ";
		cout<<"\n";

	}
	for(int j=0; j<1000; j++)
	{
		double amp = distribution(generator)*2000.+500;	
		double perc = distribution(generator);
		double amp1 = amp*perc;
		double amp2 = amp*(1.-perc);

		gen_synth_pulse(amp1, 3500,test,generator,distribution);
		add_pileup(amp2, 3500,test,generator,distribution);
		superimpose_noise(3500,test,generator,distribution);
		for(int i=0; i<3500; i++)
			cout<<test[i]<<" ";
		cout<<"\n";

	}

	auto end = chrono::steady_clock::now();	
	fprintf(stderr, "Run Time: %ld\n", chrono::duration_cast<chrono::milliseconds>(end-start).count());	
	return 0;
}
