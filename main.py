import numpy as np
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import LeaveOneOut

from data_preprocessing import IO, EventFilter, LSTMDataset
from model import FeatureExtractor, DomainClassifier, LabelPredictor,FocalLoss
from train import DomainAdaptationTrainer
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True

def cross_subject():
    
    folder='0909_overlap_epochingbyPoints'
    stride=2.5
    preictal_start_before_onset=30
    preictal_end_before_onset=5
    interictal_start_before_onset=90
    interictal_end_before_onset=30
    normalization=False
    batch_size= 32
    len_sequence=14
    features= ['SDNN', 'RMSSD','pNN50', 'TOTAL_POWER','VLF_POWER','LF_POWER', 'HF_POWER','LF_TO_HF', 'SampEn']

    ds_freq=2
    max_epochs=50
    interval=10
    hidden_size=200
    num_layers=2
    bidirectional=True
    domain_classifier_hidden=800

    all_train_patients =[10,12,15,11,16,24,26,31,34,17,18,36]
    seizureNum= [10,12,15,17,18,36]
    # loso
    loo = LeaveOneOut()
    
    for i, (train_index, test_index) in enumerate(loo.split(all_train_patients)):
        if  i==2:
            
            train_patients= [all_train_patients[i] for i in train_index]
            test_patients= [all_train_patients[i] for i in test_index]
            if test_patients[0] not in seizureNum:
                train_patients.remove(15)
                test_patients.insert(0,15)
            print('train patients', train_patients)
            print('test patients', test_patients)
            #train dataset
            train_dataset= None
            for patient in train_patients:
                print(patient)
                path= f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/feature_P{patient}.csv"
                if patient==18:
                    path= f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/rmecgoutlier_feature_P{patient}.csv" 
                HRV_dataset= IO.read_raw_csv(path, patient, folder)
                print('raw ', HRV_dataset.data.shape)
                        # normalization
                if normalization:
                    HRV_dataset =  HRV_dataset.normalization()
                    
                #eventfilter
                if patient in seizureNum:
                    event_filter= EventFilter(patient= patient, stride=stride, preictal_start_before_onset=preictal_start_before_onset,\
                                            preictal_end_before_onset=preictal_end_before_onset, \
                                            interictal_start_before_onset=interictal_start_before_onset,\
                                            interictal_end_before_onset= interictal_end_before_onset   )
                    HRV_dataset= event_filter.apply( HRV_dataset, HRV_dataset.label)
                    print('after event filter ', HRV_dataset.data.shape)
                
                else:
                    random_select_rows= np.arange(10,36,1, dtype=int)
                    HRV_dataset.apply_row_changes(random_select_rows)
                    print('nonseizure',HRV_dataset.data.shape)
                
                HRV_dataset=HRV_dataset.downsampling(sampling_class_label=0, sampling_freq=ds_freq)
                print('after ds ', HRV_dataset.data.shape)
                if train_dataset is None:
                    train_dataset= HRV_dataset
                else:
                    train_dataset.append_dataset(HRV_dataset)

            # select features
            train_dataset= train_dataset.select_features(features)
            print('new train data',train_dataset.data.to_numpy().shape,train_dataset.label.to_numpy().shape)
            train_datasets = LSTMDataset(train_dataset.data.to_numpy(),train_dataset.label.to_numpy(), len_sequence)
            train_dataloader= DataLoader(train_datasets, batch_size=batch_size, shuffle=False)


            #test scheme

            # nonseizure_test_num= [1,2,3,4,8,13,19,20,25,27,28,30,32,33]
            # for non_seizure_test in nonseizure_test_num:
            #     test_patients= [15,non_seizure_test] 
            #     print(test_patients)
            for patient in test_patients:
            
                path= f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/feature_P{patient}.csv"
                if patient==18:
                    path= f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/rmecgoutlier_feature_P{patient}.csv"
                HRV_dataset= IO.read_raw_csv(path, patient, folder)
                if normalization:
                    HRV_dataset = HRV_dataset.normalization()
                    #HRV_dataset= raw.filter_label_not_nan()
                    #HRV_dataset= HRV_dataset.filter_label_not_seizure_onset()  
                    #print('raw ', raw.data.shape)
                if patient in seizureNum:
                    event_filter= EventFilter(patient= patient, stride=stride, preictal_start_before_onset=preictal_start_before_onset,\
                                        preictal_end_before_onset=preictal_end_before_onset, \
                                        interictal_start_before_onset=interictal_start_before_onset,\
                                        interictal_end_before_onset= interictal_end_before_onset   )
                    HRV_dataset= event_filter.apply(HRV_dataset, HRV_dataset.label)
                    print('after event filter ', HRV_dataset.data.shape)
                    HRV_dataset=HRV_dataset.downsampling(sampling_class_label=0, sampling_freq=ds_freq)
                    print('after ds ', HRV_dataset.data.shape)
            
                if patient== test_patients[0]:
                    test_dataset= HRV_dataset
                else:

                    test_dataset.append_dataset(HRV_dataset)


            test_dataset= test_dataset.select_features(features)
            test_datasets = LSTMDataset(test_dataset.data.to_numpy(),test_dataset.label.to_numpy(), len_sequence)
            test_dataloader= DataLoader(test_datasets, batch_size=batch_size, shuffle=False)

          
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            if bidirectional:
                bi=2
            else:
                bi=1
            feature_extractor = FeatureExtractor(input_size= train_dataset.data.shape[1], hidden_size= hidden_size, num_layers= num_layers, bidirectional= bidirectional).to(device)
            label_predictor = LabelPredictor(input_size=hidden_size*num_layers*bi, hidden_size=  domain_classifier_hidden).to(device)
            domain_classifier = DomainClassifier(input_size=hidden_size*num_layers*bi, hidden_size= domain_classifier_hidden).to(device)
            #10,: 4/8, 12: 1.5/4
            class_criterion= FocalLoss(alpha=1, gamma=2, reduction='mean')
            #domain_criterion= FocalLoss(alpha=1, gamma=4, reduction='sum')
            domain_criterion=  nn.BCEWithLogitsLoss()
            optimizer_F = torch.optim.Adam(feature_extractor.parameters())
            optimizer_C = torch.optim.Adam(label_predictor.parameters())
            optimizer_D = torch.optim.Adam(domain_classifier.parameters())
            # Initialize trainer
            trainer = DomainAdaptationTrainer(feature_extractor, label_predictor, domain_classifier, \
                        optimizer_F, optimizer_C, optimizer_D, \
                        train_dataloader,test_dataloader, max_epochs, interval, device,\
                            domain_criterion, class_criterion, patient)

            # Train and evaluate
            trainer.train()
            trainer.evaluate(test_dataloader, test_patients)



if __name__ == "__main__":   
    
    cross_subject()