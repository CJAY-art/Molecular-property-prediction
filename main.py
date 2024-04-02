"""run model train/test/interpret on a dataset."""

#from chemprop.train import chemprop_train
# from chemprop.train import runing_eval
from chemprop.sklearn_train import sklearn_train
#from chemprop.interpret import chemprop_interpret


if __name__ == '__main__':
    #chemprop_train()
    # runing_eval()
    #chemprop_interpret()
    sklearn_train()