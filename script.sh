#python3 centralized_run.py --Dataset=$1 --strategy=FedAvg --nbr_clients=$2 --nbr_rounds=$3 --accumulated_data=False --centralized_percentage=1 --directory_name=$4

python3 federated_run.py --Dataset=$1 --strategy=FedAvg --nbr_clients=$2 --nbr_rounds=$3 --accumulated_data=False  --directory_name=$4
