# Internship - Federated Learning Benchmark

### Motives

Generate benchmark to compare the performance of federated learning and centralized learning on various Dataset

### Dataset

:white_check_mark: MNIST <br/>
:white_check_mark: CIFAR10 <br/>
:white_check_mark: DisasterTweets <br/>
:white_check_mark: BigQoe (=JS) <br/>

### Add your own dataset :

To add your own dataset to test, you have to follow the next two steps.

- **Steps 1** : Add your data folder

data
|
-> Create Folder : data*"name_of_your_dataset"
|
-> Create File : Preprocessing*"name*of_your_dataset".py
|
-> Create Function : Preprocessing*"name_of_your_dataset".py : you need a function called load_data\*"name_of_your_dataset" that will
return X_train, X_test, y_train, y_test.

- **Steps 2** : Add your model file

model
|
-> Create model*"name_of_your_dataset".py
|
-> create_model*"name_of_your_dataset"

### How to run it 

python3 Launcher.py --Dataset="Dataset" --strategy="stategy" --nbr_rounds="nbr-rounds" --nbr_clients="nbr-clients" --centralized_percentage="%" --accumulated_data="True/False"
