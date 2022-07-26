

for i in 1 2 3 4 5
do
	for strategy in FedAvg FedYogi FedAdam FedAdagrad
	do
		python3 Launcher.py --Dataset=IMDB --strategy=$strategy --nbr_clients=20 --nbr_rounds=100 
	done
done



'
for i in 2 3 4 5
do
	for strategy in FedAvg FedYogi FedAdam FedAdagrad
	do
		python3 Launcher.py --Dataset=CIC_IDS2017 --strategy=$strategy --nbr_clients=8 --nbr_rounds=5 
done
done
'





'
for i in 1 2 3 4 5
do
	for strategy in FedAvg FedYogi FedAdam FedAdagrad
	do
		python3 Launcher.py --Dataset=DisasterTweets --strategy=$strategy --nbr_clients=5 --nbr_rounds=100 
	done
done


for i in 1 2 3 4 5
do
	for strategy in FedAvg FedYogi FedAdam FedAdagrad
	do
		python3 Launcher.py --Dataset=MNIST --strategy=$strategy --nbr_clients=7 --nbr_rounds=50 
	done
done



for i in 1 2 3 4 5
do
	for strategy in FedAvg FedYogi FedAdam FedAdagrad
	do
		python3 Launcher.py --Dataset=JS --strategy=$strategy --nbr_clients=20 --nbr_rounds=100 
	done
done
'

