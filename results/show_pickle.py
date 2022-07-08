import pickle 

client = ["client_number_"+ str(i) for i in range(4)]
for i in ["server","centralized"]+client:
    print(i)
    f = open(i,"rb")
    metrics = pickle.load(f)
    try :
        time = pickle.load(f)
        print(i, metrics, time)
    except:
        print(i, metrics)
