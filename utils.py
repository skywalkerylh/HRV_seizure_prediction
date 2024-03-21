import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

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
    senstivity,specificity=0,0
    acc= len(np.where(ypred==ytrue)[0])/len(ytrue)
    print('acc',round(acc,2))
    if not np.all(ytrue==0) and not np.all(ytrue==1):

        tn, fp, fn, tp = confusion_matrix(ytrue, ypred).ravel()
        specificity = tn / (tn+fp)
        senstivity= tp / (tp+fn)
        print('sensitivitiy:', round(senstivity,2))
        print('specificity:',round( specificity,2))
      
    return acc, senstivity, specificity
    
def metrics_individual(ytrue, ypred):
    change_idx= np.where(ytrue==1)[0][-1]
    #print(change_idx)
    print('first subject')
    acc1, senstivity1, specificity1= metrics(ytrue[:change_idx], ypred[:change_idx])
    print('second subject')
    acc2, senstivity2, specificity2= metrics(ytrue[change_idx+1:], ypred[change_idx+1:])
    print('2nd ytrue',ytrue[change_idx+1:])
    print('2nd ypred',ypred[change_idx+1:])
    return acc1, senstivity1, specificity1, acc2, senstivity2, specificity2

def record_metrics(acc_list, sensitivity_list, specificity_list, \
                   acc_value, sensitivity_value, specificity_value):
    acc_list.append(acc_value)
    sensitivity_list.append(sensitivity_value)
    specificity_list.append(specificity_value)
    return acc_list, sensitivity_list, specificity_list

def gen_pred_info(pred_labels,dataset, len_sequence, patient):
    
    # remove head n len_sequence
    dataset= dataset.apply_row_changes(np.arange(len_sequence-1,len(dataset.label),1))
    # select 2nd subject ypred
    change_idx= np.where(dataset.label.values==1)[0][-1]
    pred_labels= pred_labels[change_idx+1:]
    pred_labels= pd.DataFrame(pred_labels, columns= ['prediction'])
    
    # select 2nd subject ytrue
    dataset= dataset.apply_row_changes(np.arange(change_idx+1, len(dataset.label),1))
    assert len(dataset.label) == len(pred_labels)
    
    # get info 
    dataset.data['patient']= patient
    pred_labels.index= dataset.data.index
    pred_info= pd.concat([dataset.data, dataset.info, pred_labels], axis=1)
    print('pred info', pred_info.shape)

    return pred_info