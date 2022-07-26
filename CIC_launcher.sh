#!/bin/bash


for i in 1 2 3 4 5
do
	for nbr_clients in 2 4 6 8 10 12
	do
		for strategy in FedAvg FedYogi FedAdam FedAdagrad
		do
			for nbr_rounds in 1 2 3 4 5
			do
				python3 Launcher.py --Dataset=CIC_IDS2017 --strategy=$strategy --nbr_clients=$nbr_clients --nbr_rounds=$nbr_rounds
			done
		done
	done
done

