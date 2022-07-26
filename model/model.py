


class FLModel() :
    def __init__(self,dataset):

        if dataset == "IMDB":
            from .model_IMDB import create_model_IMDB
            self.model, self.loss, self.optimizer, self.metrics  = create_model_IMDB()

        if dataset == "MNIST":
            print("Loading model")
            from .model_MNIST import create_model_MNIST
            self.model, self.loss, self.optimizer, self.metrics  = create_model_MNIST()
            print("Model loaded")
        if dataset == "DisasterTweets": 
            from .model_DisasterTweets import create_model_DisasterTweets
            self.model,self.loss, self.optimizer,self.metrics = create_model_DisasterTweets()
        
        elif dataset == "Bostonhouse" :
            from .model_Bostonhouse import create_model_Bostonhouse
            self.model,self.loss, self.optimizer, self.metrics = create_model_Bostonhouse()

        elif dataset == "CIC_IDS2017":
            from .model_CIC_IDS2017 import create_model_CIC_IDS2017 
            self.model, self.loss, self.optimizer, self.metrics= create_model_CIC_IDS2017()

        elif dataset == "CIFAR10":
            from .model_CIFAR10 import create_model_CIFAR10
            self.model, self.loss, self.optimizer, self.metrics= create_model_CIFAR10()
        elif dataset == "JS":
            from .model_JS import create_model_JS
            self.model, self.loss, self.optimizer, self.metrics= create_model_JS()

    
