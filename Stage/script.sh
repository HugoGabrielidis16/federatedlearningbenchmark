

python3 Launcher.py --Dataset=MNIST --strategy=FedAvg --nbr_clients=10 --nbr_rounds=100 --accumulated_data=True --centralized_percentage=1
python3 Launcher.py --Dataset=MNIST --strategy=FedAdam --nbr_clients=10 --nbr_rounds=100 --accumulated_data=True --centralized_percentage=1
python3 Launcher.py --Dataset=MNIST --strategy=FedAvg --nbr_clients=10 --nbr_rounds=100 --accumulated_data=True --centralized_percentage=1


python3 Launcher.py --Dataset=CIFAR10 --strategy=FedAvg --nbr_clients=7 --nbr_rounds=50 --accumulated_data=True --centralized_percentage=1
python3 Launcher.py --Dataset=CIFAR10 --strategy=FedAdam --nbr_clients=7 --nbr_rounds=50 --accumulated_data=True --centralized_percentage=1
python3 Launcher.py --Dataset=CIFAR10 --strategy=FedYogi --nbr_clients=7 --nbr_rounds=50 --accumulated_data=True --centralized_percentage=1

python3 Launcher.py --Dataset=DisasterTweets --strategy=FedAvg --nbr_clients=5 --nbr_rounds=50 --accumulated_data=True --centralized_percentage=1
python3 Launcher.py --Dataset=DisasterTweets --strategy=FedAdam --nbr_clients=5 --nbr_rounds=50 --accumulated_data=True --centralized_percentage=1
python3 Launcher.py --Dataset=DisasterTweets --strategy=FedYogi --nbr_clients=5 --nbr_rounds=50 --accumulated_data=True --centralized_percentage=1

python3 Launcher.py --Dataset=IMDB --strategy=FedAdam --nbr_clients=5 --nbr_rounds=50 --accumulated_data=True --centralized_percentage=1
python3 Launcher.py --Dataset=IMDB --strategy=FedYogi --nbr_clients=5 --nbr_rounds=50 --accumulated_data=True --centralized_percentage=1
python3 Launcher.py --Dataset=IMDB --strategy=FedYogi --nbr_clients=5 --nbr_rounds=50 --accumulated_data=True --centralized_percentage=1
