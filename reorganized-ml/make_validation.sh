#!/bin/bash
cd ~/Documents/ML/
for i in {1..101}
do
	data_simulators/validation/bin/pulse_generation data/val$i.dat
done

