import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from data_preprocessing import IO, LSTMDataset, preprocessing_pipeline
from model import FeatureExtractor, DomainClassifier, LabelPredictor,FocalLoss, LSTM, generate_LOSO_train_test_subjects
from train import DomainAdaptationTrainer, LSTMTrainer
from utils import record_metrics

folder='0909_overlap_epochingbyPoints'
normalization=False
batch_size= 32
len_sequence=14
interval=20
ds_freq=2
seizureNum= [10,12,15,17,18,36]
features= ['SDNN', 'RMSSD','pNN50', 'TOTAL_POWER','VLF_POWER','LF_POWER', 'HF_POWER','LF_TO_HF', 'SampEn']

def internal_validation():
    train_patients =[10,12,11,16,24,26,31,34,36]
    #train_patients=[10,12,11,21,9,35,36]
    max_epochs=100
    hidden_size=200
    num_layers=2
    bidirectional=True
    domain_classifier_hidden=800
    all_acc1, all_sensitivity1, all_specificity1=[],[],[]
    all_acc2, all_sensitivity2, all_specificity2=[],[],[]

    #train dataset
    print('train patients: ',train_patients)
    train_dataset=None
    for patient in train_patients:
        path= f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/feature_P{patient}.csv"
        if patient==18:
            path= f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/rmecgoutlier_feature_P{patient}.csv" 
        HRV_dataset= IO.read_raw_csv(path, patient, folder)
        #print('raw ', HRV_dataset.data.shape)
        HRV_dataset= preprocessing_pipeline(HRV_dataset, normalization, seizureNum, patient, ds_freq)
        # stack data
        if train_dataset is None:
            train_dataset= HRV_dataset
        else:
            train_dataset.append_dataset(HRV_dataset)

    train_dataset= train_dataset.select_features(features)
    print('new train data',train_dataset.data.to_numpy().shape,train_dataset.label.to_numpy().shape)
    train_datasets = LSTMDataset(train_dataset.data.to_numpy(),train_dataset.label.to_numpy(), len_sequence)
    train_dataloader= DataLoader(train_datasets, batch_size=batch_size, shuffle=False)
    
    #test scheme
    nonseizure_test_num= [29,31,16,34,24,26,22]
    #nonseizure_test_num= [9,21,22,29,35]
    for non_seizure_test in nonseizure_test_num:
        test_patients= [15,non_seizure_test] 
        print('test patients: ',test_patients)
        acc_list1, sensitivity_list1, specificity_list1=[],[],[]
        acc_list2, sensitivity_list2, specificity_list2=[],[],[]
        for seed in range(3):
            torch.manual_seed(seed)
            # test dataset
            test_dataset=None
            for patient in test_patients:
                path= f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/feature_P{patient}.csv"
                if patient==18:
                    path= f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/rmecgoutlier_feature_P{patient}.csv" 
                HRV_dataset= IO.read_raw_csv(path, patient, folder)
                #print('raw ', HRV_dataset.data.shape)
                HRV_dataset= preprocessing_pipeline(HRV_dataset, normalization, seizureNum, patient, ds_freq)
                
                # stack data
                if test_dataset is None:
                    test_dataset= HRV_dataset
                else:
                    test_dataset.append_dataset(HRV_dataset)

            test_dataset= test_dataset.select_features(features)
            print('new train data',test_dataset.data.to_numpy().shape,test_dataset.label.to_numpy().shape)
            test_datasets = LSTMDataset(test_dataset.data.to_numpy(),test_dataset.label.to_numpy(), len_sequence)
            test_dataloader= DataLoader(test_datasets, batch_size=batch_size, shuffle=False)
            
        
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            if bidirectional:
                bi=2
            else:
                bi=1
            feature_extractor = FeatureExtractor(input_size=len(features), hidden_size= hidden_size, num_layers= num_layers, bidirectional= bidirectional).to(device)
            label_predictor = LabelPredictor(input_size=hidden_size*num_layers*bi, hidden_size=  domain_classifier_hidden).to(device)
            domain_classifier = DomainClassifier(input_size=hidden_size*num_layers*bi, hidden_size= domain_classifier_hidden).to(device)
            class_criterion= FocalLoss(alpha=1, gamma=2, reduction='mean')
            
            domain_criterion=  nn.BCEWithLogitsLoss()
            optimizer_F = torch.optim.Adam(feature_extractor.parameters())
            optimizer_C = torch.optim.Adam(label_predictor.parameters())
            optimizer_D = torch.optim.Adam(domain_classifier.parameters())
            # Initialize trainer
            trainer = DomainAdaptationTrainer(feature_extractor, label_predictor, domain_classifier, \
                        optimizer_F, optimizer_C, optimizer_D, \
                        train_dataloader,test_dataloader, max_epochs, interval, device,\
                            domain_criterion, class_criterion, patient= test_patients[0])

            # Train and evaluate
            trainer.train()
            vars = trainer.evaluate(test_dataloader, test_patients)
            acc_list1, sensitivity_list1, specificity_list1= record_metrics(acc_list1, sensitivity_list1, specificity_list1, \
                            vars.acc1, vars.sensitivity1, vars.specificity1)
            acc_list2, sensitivity_list2, specificity_list2= record_metrics(acc_list2 ,sensitivity_list2, specificity_list2, \
                            vars.acc2, vars.sensitivity2, vars.specificity2)

        
        all_acc1, all_sensitivity1, all_specificity1= record_metrics(all_acc1, all_sensitivity1, all_specificity1, \
                        round(np.mean(acc_list1),2),round(np.mean(sensitivity_list1),2), round(np.mean(specificity_list1),2) )
        all_acc2, all_sensitivity2, all_specificity2= record_metrics(all_acc2, all_sensitivity2, all_specificity2, \
                        round(np.mean(acc_list2),2),round(np.mean(sensitivity_list2),2), round(np.mean(specificity_list2),2) )

        print('all acc1', all_acc1)
        print('all sen1', all_sensitivity1)
        print('all speci1', all_specificity1)
        print('all acc2', all_acc2)
        print('all sen2', all_sensitivity2)
        print('all speci2', all_specificity2)

def LOSO():
    #all_train_patients= [10,12,15,11,16,24,26,31,34,17,18,36]
    all_train_patients=[10,12,15,11,21,9,35,36]
    max_epochs=100
    hidden_size=200
    num_layers=2
    bidirectional=True
    domain_classifier_hidden=800

    all_acc, all_sensitivity, all_specificity=[],[],[]

    for run_index in range(len(all_train_patients)):
        # split train test subjects
        train_patients, test_patients= generate_LOSO_train_test_subjects(run_index, all_train_patients,seizureNum)
        acc, sensitivity, specificity=[],[],[]
        # repetitive exp
        for seed in range(3):
            torch.manual_seed(seed)

            #train dataset
            train_dataset=None
            for patient in train_patients:
                path= f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/feature_P{patient}.csv"
                if patient==18:
                    path= f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/rmecgoutlier_feature_P{patient}.csv" 
                HRV_dataset= IO.read_raw_csv(path, patient, folder)
                #print('raw ', HRV_dataset.data.shape)
                HRV_dataset= preprocessing_pipeline(HRV_dataset, normalization, seizureNum, patient, ds_freq)
                # stack data
                if train_dataset is None:
                    train_dataset= HRV_dataset
                else:
                    train_dataset.append_dataset(HRV_dataset)

          
            train_dataset= train_dataset.select_features(features)
            print('new train data',train_dataset.data.to_numpy().shape,train_dataset.label.to_numpy().shape)
            train_datasets = LSTMDataset(train_dataset.data.to_numpy(),train_dataset.label.to_numpy(), len_sequence)
            train_dataloader= DataLoader(train_datasets, batch_size=batch_size, shuffle=False)
            
            # test dataset
            test_dataset=None
            for patient in test_patients:
                path= f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/feature_P{patient}.csv"
                if patient==18:
                    path= f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/rmecgoutlier_feature_P{patient}.csv" 
                HRV_dataset= IO.read_raw_csv(path, patient, folder)
                #print('raw ', HRV_dataset.data.shape)
                HRV_dataset= preprocessing_pipeline(HRV_dataset, normalization, seizureNum, patient, ds_freq)

                # stack data
                if test_dataset is None:
                    test_dataset= HRV_dataset
                else:
                    test_dataset.append_dataset(HRV_dataset)

            test_dataset= test_dataset.select_features(features)
            print('new train data',test_dataset.data.to_numpy().shape,test_dataset.label.to_numpy().shape)
            test_datasets = LSTMDataset(test_dataset.data.to_numpy(),test_dataset.label.to_numpy(), len_sequence)
            test_dataloader= DataLoader(test_datasets, batch_size=batch_size, shuffle=False)
           
            # model setting
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            if bidirectional:
                bi=2
            else:
                bi=1
            feature_extractor = FeatureExtractor(input_size=len(features), hidden_size= hidden_size, num_layers= num_layers, bidirectional= bidirectional).to(device)
            label_predictor = LabelPredictor(input_size=hidden_size*num_layers*bi, hidden_size=  domain_classifier_hidden).to(device)
            domain_classifier = DomainClassifier(input_size=hidden_size*num_layers*bi, hidden_size= domain_classifier_hidden).to(device)
            class_criterion= FocalLoss(alpha=1, gamma=2, reduction='mean')
            
            domain_criterion=  nn.BCEWithLogitsLoss()
            optimizer_F = torch.optim.Adam(feature_extractor.parameters())
            optimizer_C = torch.optim.Adam(label_predictor.parameters())
            optimizer_D = torch.optim.Adam(domain_classifier.parameters())
            # Initialize trainer
            trainer = DomainAdaptationTrainer(feature_extractor, label_predictor, domain_classifier, \
                        optimizer_F, optimizer_C, optimizer_D, \
                        train_dataloader,test_dataloader, max_epochs, interval, device,\
                            domain_criterion, class_criterion, patient= test_patients[0])

            # Train and evaluate
            trainer.train()
            vars = trainer.evaluate(test_dataloader, test_patients)
            acc, sensitivity, specificity= record_metrics(acc, sensitivity, specificity, \
                           vars.acc, vars.sensitivity, vars.specificity)

        
        all_acc, all_sensitivity, all_specificity= record_metrics(all_acc, all_sensitivity, all_specificity, \
                       round(np.mean(acc),2),round(np.mean(sensitivity),2), round(np.mean(specificity),2) )

        print('all acc', all_acc)
        print('all sen', all_sensitivity)
        print('all speci', all_specificity)

def individual_subject():
    
    
    stride=2.5
    preictal_start_before_onset=30
    preictal_end_before_onset=5
    interictal_start_before_onset=90
    interictal_end_before_onset=30
    normalization=False

    #features=[ 'SDNN','RMSSD','RRMean','ApEn', 'SampEn']
   
    max_epochs=100
  
    hidden_size=400
    num_layers=2
    bidirectional=True
 
    train_ratio=0.4
    val_ratio=0.3
    lr = 0.00005
    acc, sensitivity, specificity=[],[],[]
    for seed in range(2):
        for patient in [15]:
            torch.manual_seed(seed)
            print(patient)
            path= f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/feature_P{patient}.csv"
            if patient==18:
                path= f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/rmecgoutlier_feature_P{patient}.csv" 
            HRV_dataset= IO.read_raw_csv(path, patient, folder)
            print('raw ', HRV_dataset.data.shape)
                    # normalization
            HRV_dataset= preprocessing_pipeline(HRV_dataset, normalization, seizureNum, patient, ds_freq)
            # select features
            HRV_dataset= HRV_dataset.select_features(features)
            # split dataset 
            train_data, train_label, val_data, val_label, test_data, test_label = HRV_dataset.split(train_ratio,val_ratio)
            #class weight
            class_weights=  (train_label==1).sum()/((train_label==1).sum()+(train_label==0).sum())
            print('class proportion ', class_weights)
            # 
            train_dataset = LSTMDataset(train_data.to_numpy(),train_label.to_numpy(), len_sequence)
            train_dataloader= DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            val_dataset = LSTMDataset(val_data.to_numpy(),val_label.to_numpy(), len_sequence)
            val_dataloader= DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_dataset = LSTMDataset(test_data.to_numpy(),test_label.to_numpy(), len_sequence)
            test_dataloader= DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            print('train data', train_data.shape)
            print('val data', val_data.shape)
            print('test data', test_data.shape)
            
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            model = LSTM(input_size=train_data.shape[1], hidden_size=hidden_size, output_size=2, num_layer=num_layers, bidirectional=bidirectional)
            criterion= FocalLoss(alpha=1, gamma=2, reduction='mean')
            optimizer=torch.optim.Adam(model.parameters(), lr=lr)
            # Initialize trainer
            trainer= LSTMTrainer(patient,  model, train_dataloader, val_dataloader, test_dataloader, \
                    criterion, optimizer, device, max_epochs, interval)
            # Train and evaluate
            trainer.train()
            vars= trainer.evaluate()
            acc.append(vars.acc)
            sensitivity.append(vars.sensitivity)
            specificity.append(vars.specificity)

    print('averaged acc',round(np.mean(acc),2), '\t', round(np.std(acc),2))
    print('sen acc',round(np.mean(sensitivity),2), '\t', round(np.std(sensitivity),2))
    print('speci acc',round(np.mean(specificity),2), '\t', round(np.std(specificity),2))
    

if __name__ == "__main__":   
    #individual_subject()
    #LOSO()
    internal_validation()