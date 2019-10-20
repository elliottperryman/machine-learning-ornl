#!/bin/bash
cd ~/Documents/ML/
for i in {1..101}
do
	data_simulators/training/bin/pulse_generation data/$i.dat
done

