#!/bin/bash

for i in range 5
do
	for dataset in CIC_IDS2017 
	do
		for nbr_clients in 8
		do
			for strategy in FedAvg FedYogi FedAdam FedAdagrad
			do
				for nbr_rounds in  5
				do
					python3 Launcher.py --Dataset=$dataset --strategy=$strategy --nbr_clients=$nbr_clients --nbr_rounds=$nbr_rounds
				done	
			done
		done
	done
done


