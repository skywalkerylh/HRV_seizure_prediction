import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
class IO:
    def __init__(self):
        pass
    def read_raw_csv(path, patient, folder):
        return RawDataset(path, patient, folder)
    
class RawDataset:
    def __init__(self, path, patient, folder):
        self.data = pd.read_csv(path) # 可放到read_raw_csv
        self.label= self.data['label']
        self.info= self.data.loc[:,['startime', 'endtime','file','day']]
        not_hrv_columns= ['label', 'startime', 'endtime','file','day']
        self.data= self.data.loc[:,~self.data.columns.isin(not_hrv_columns)]
        
        self.patient= [patient]
        self.folder= folder
    def labelisnan(self):
        return self.label.isnull().sum() == self.label.shape[0]
    
    def downsampling(self, sampling_class_label, sampling_freq):
        
        if self.labelisnan==True:
            print('labelisnan')
            return self

            
        class_label_idx =self.label[self.label==sampling_class_label].index.tolist()
        ds_class_label_idx= class_label_idx[:: sampling_freq]
        other_class_label_idx =self.label[self.label!= sampling_class_label].index.tolist()
        label_idx =np.concatenate((ds_class_label_idx, other_class_label_idx))
        label_idx = np.sort(label_idx)
    
        self.apply_row_changes(label_idx)

        return self
       

    
    def select_features(self, features):
        self.data= self.data.loc[:,features]
        return self
    
    def filter_label_not_nan(self):
        not_nan_bool= ~self.label.isna()
        self.apply_row_changes(not_nan_bool)
        return self
    
    def filter_label_not_seizure_onset(self):
        not_seizure_onset_bool= self.label!=2
        self.apply_row_changes(not_seizure_onset_bool)
        return self
    
    def normalization(self):
        #scaler = MinMaxScaler()
        #self.data= scaler.fit_transform(self.data)
        def standardize_column(column):
            mean_value = column.mean()
            std_value = column.std()
            standardized_column = (column - mean_value) / std_value
            return standardized_column
        
        # Apply standardization to each column
        self.data = self.data.apply(standardize_column, axis=0)
        print(self.data['RMSSD'].mean())
        return self
    
    def apply_row_changes(self, criteria):
        #print(criteria.dtype)
        if criteria.dtype== 'bool':
            
            self.data= self.data[criteria]
            self.label= self.label[criteria]
            self.info= self.info[criteria]
        
        elif criteria.dtype=='int64' or criteria.dtype== 'int32':
           
            self.data= self.data.iloc[criteria]
            self.label= self.label.iloc[criteria]
            self.info= self.info.iloc[criteria]
           
    def append_dataset(self, dataset):
        self.data= pd.concat([self.data, dataset.data], ignore_index=True)
        self.label= pd.concat([self.label, dataset.label], ignore_index=True)
        self.info= pd.concat([self.info, dataset.info], ignore_index=True)
        self.patient= np.concatenate((self.patient, dataset.patient))

        self.data= self.data.reset_index(drop=True)
        self.label= self.label.reset_index(drop=True)
        self.info= self.info.reset_index(drop=True)
        return self
    

class EventFilter():
    def __init__(self, patient, stride, \
                 preictal_start_before_onset, preictal_end_before_onset,\
                 interictal_start_before_onset,  interictal_end_before_onset ):
        self.patient= patient
        self.stride= stride
        self.preictal_start_before_onset = preictal_start_before_onset
        self.preictal_end_before_onset=preictal_end_before_onset
        self.interictal_start_before_onset=interictal_start_before_onset
        self.interictal_end_before_onset=interictal_end_before_onset

    def find_event_idx_labels(self, labels):
        prev=-1
        start_idx= []
        for idx, label in enumerate(labels):
            if np.isnan(label):
                break
            if label!= prev: 
                start_idx.append([idx,label])
            prev= label

        return np.array(start_idx).astype(int)  

    def select_preictal_interictal_idx(self, start_idx):
        target_label=0
        idx_of_target_label= np.where(start_idx[:,1]== target_label)[0]

        selected_idx= list()
        
        for idx in idx_of_target_label:
            if idx== len(start_idx)-1:
                break 
         
            label0_startidx, label1_startidx, label2_startidx=start_idx[idx,0],start_idx[idx+1,0], start_idx[idx+2,0]
            len_label0= label1_startidx - label0_startidx
          
            preictal_start_idx= label2_startidx - (self.preictal_start_before_onset/self.stride)
            preictal_end_idx= label2_startidx - (self.preictal_end_before_onset/self.stride)
            interictal_start_idx= label2_startidx - (self.interictal_start_before_onset/self.stride)
            interictal_end_idx= label2_startidx - (self.interictal_end_before_onset/self.stride)

            len_interictal= interictal_end_idx - interictal_start_idx
            
            if len_label0 < len_interictal: 
                
                interictal_start_idx= label0_startidx
            if preictal_start_idx < label1_startidx:
                
                preictal_start_idx= label1_startidx
                interictal_end_idx+=(label1_startidx - preictal_start_idx)+1
                if self.patient== 12 or self.patient ==15: 
                    print('interictal start+1')
                    interictal_start_idx+=1
            if interictal_start_idx < label0_startidx:
                interictal_start_idx= label0_startidx

            event_preictal_idx= range(int(preictal_start_idx), int(preictal_end_idx))
            event_interictal_idx= range(int(interictal_start_idx),int( interictal_end_idx))
            #print(event_interictal_idx, event_preictal_idx )
            selected_idx+= event_interictal_idx
            selected_idx+= event_preictal_idx
       
            
        return selected_idx
    
    def apply(self,raw, labels):
        start_idx=self.find_event_idx_labels(labels)
        #print(start_idx)
        selected_idx=self.select_preictal_interictal_idx(start_idx)
       
        raw.data= raw.data.iloc[selected_idx].reset_index(drop=True)
        raw.label= raw.label.iloc[selected_idx].reset_index(drop=True)
        raw.info= raw.info.iloc[selected_idx].reset_index(drop=True)
        return raw

class LSTMDataset(Dataset):
    def __init__(self, data, labels,len_sequence):
        self.data, self.labels= self.create_sequence(data,labels,len_sequence)
        
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels.reshape(-1), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_sample = self.data[index]
        label_sample = self.labels[index]
        return data_sample, label_sample
    
    def create_sequence(self,data, labels, T):
        X, y = [], []
        for i in range(labels.shape[0] - (T-1)):
            X.append(data[i:i+T])
            y.append(labels[i + (T-1)])
        X, y = np.array(X), np.array(y).reshape(-1,1)
     

        return X,y




