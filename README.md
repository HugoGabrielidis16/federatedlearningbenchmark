# Internship - Federated Learning Benchmark

### Context

In the context of my 6 month internship at Orange Innovation, my tutor and I focused on comparing federated and centralized Learning.
Federated Learning is an apporach where instead of sending your data to the cloud where there could be confidentiality issue, you have the model on your own
devices which is updated each "rounds" using everyone weights.
This approach was created in 2017 by Google and has the advantages of being safer, doesn't require a constant connection, less latency.
During my internship, we concentrated on seeing if all those advantages came with a trade off in performace.

### Results

After experimentation we saw that if using the same model for both centralized & federated we achieved the same performance for both.
And while the centralized approach was indeed faster it was still in the same order of magnitude.

### Datasets

:white_check_mark: MNIST <br/>
:white_check_mark: CIFAR10 <br/>
:white_check_mark: DisasterTweets <br/>
:white_check_mark: BigQoe (=JS) <br/>

### Add your own dataset :

To add your own dataset to test, you have to follow the next two steps.

- **Steps 1** : Create your data folder

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
