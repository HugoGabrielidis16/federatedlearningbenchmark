def run_centralized_Shakespeare(epochs, directory_name):

    import pickle
    from Model.model_Shakespeare import create_model_Shakespeare
    from data.data_Shakespeare.Preprocessing_Shakespeare import dataset

    model = create_model_Shakespeare()
    history = model.fit(dataset, epochs=epochs, verbose=1)

    dict = history.history
    list = []
    for keys in list(dict.keys()):
        if "val" in keys and "loss" not in keys:
            for i in range(len(keys)):
                list.append((dict[keys][i], dict[keys][i]))

    file_name = directory_name + "/centralized"
    with open(file_name, "wb") as f:
        pickle.dump(list, f)
