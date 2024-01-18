import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def plot( history_train, history_test,patient):
        fig, axes = plt.subplots(ncols=2, figsize=(9, 4.5))
        for ax, metric in zip(axes, ['loss', 'acc']):
            ax.plot(history_train[metric])
            ax.plot(history_test[metric])
            ax.set_xlabel('epoch', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.legend(['Train', 'Test'], loc='best')
        fig.suptitle(f'P{patient}') 
        plt.show()
        
def metrics(ytrue, ypred):
    acc= len(np.where(ypred==ytrue)[0])/len(ytrue)
    print('acc',acc)
    if not np.all(ytrue==0):

        tn, fp, fn, tp = confusion_matrix(ytrue, ypred).ravel()
        fpr= fp/ (fp+tn)
        
        specificity = tn / (tn+fp)
        senstivity= tp / (tp+fn)

    
        print('sensitivitiy:', senstivity)
        print('specificity:', specificity)
        print('fpr: ',fpr)
    
def metrics_individual(ytrue, ypred):
    change_idx= np.where(ytrue==1)[0][-1]
    print(change_idx)
    print('first subject')
    metrics(ytrue[:change_idx], ypred[:change_idx])
    print('second subject')
    metrics(ytrue[change_idx:], ypred[change_idx:])