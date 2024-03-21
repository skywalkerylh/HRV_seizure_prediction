import torch
import numpy as np
from utils import metrics, metrics_individual, plot
import torch.nn.functional as F

class DomainAdaptationTrainer:
    def __init__(self, feature_extractor, label_predictor, domain_classifier, \
                 optimizer_F, optimizer_C, optimizer_D, \
                dataloader_source, dataloader_target, max_epochs, interval, device,\
                domain_criterion, class_criterion, patient):
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor
        self.domain_classifier = domain_classifier
        self.dataloader_source = dataloader_source
        self.dataloader_target = dataloader_target
        self.optimizer_F= optimizer_F
        self.optimizer_C= optimizer_C
        self.optimizer_D= optimizer_D
        self.max_epochs = max_epochs
        self.interval = interval
        self.device= device
        self. domain_criterion= domain_criterion
        self.class_criterion= class_criterion
        self.patient= patient
        # Other initialization code for criterion and optimizers here

    def train(self):
        # Training loop code here
        for epoch in range(self.max_epochs):
            '''
            Args:
            source_dataloader: source data的dataloader
            target_dataloader: target data的dataloader
            lamb: 調控adversarial的loss係數。
        '''
            lamb=0.1
            #lamb=np.log(1.02+1.7*epoch/self.max_epochs)
            # D loss: Domain Classifier的loss
            # F loss: Feature Extrator & Label Predictor的loss
            # total_hit: 計算目前對了幾筆 total_num: 目前經過了幾筆
            running_D_loss, running_F_loss, min_loss = 0.0, 0.0, 1e10
            total_hit, total_num = 0.0, 0.0

            for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(self.dataloader_source, self.dataloader_target)):

                source_data = source_data.cuda()
                source_label=F.one_hot(source_label, num_classes=2)
                source_label= source_label.type(torch.FloatTensor).cuda()
                #source_label = source_label.cuda()
                target_data = target_data.cuda()
                
                # 我們把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
                mixed_data = torch.cat([source_data, target_data], dim=0)
                #domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1], dtype=torch.int64).cuda()
                domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0]], dtype=torch.long)
                # 設定source data的label為1
                domain_label[:source_data.shape[0]] = 1
                domain_label=F.one_hot(domain_label, num_classes=2)
                domain_label= domain_label.type(torch.FloatTensor).cuda()
               
                # Step 1 : 訓練Domain Classifier
                feature_extractor = self.feature_extractor(mixed_data)
                # 因為我們在Step 1不需要訓練Feature Extractor，所以把feature detach避免loss backprop上去。
                domain_logits = self.domain_classifier(feature_extractor.detach())
                
                d_loss = self.domain_criterion(domain_logits, domain_label)
                running_D_loss+= d_loss.item()
                d_loss.backward()
                self.optimizer_D.step()

                # Step 2 : 訓練Feature Extractor和Domain Classifier
                class_logits = self.label_predictor(feature_extractor[:source_data.shape[0]])
                domain_logits = self.domain_classifier(feature_extractor)
                # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。
               
                f_loss = self.class_criterion(class_logits, source_label) - lamb * self.domain_criterion(domain_logits, domain_label)
                
                running_F_loss+= f_loss.item()
                f_loss.backward()
                self.optimizer_F.step()
                self.optimizer_C.step()

                self.optimizer_D.zero_grad()
                self.optimizer_F.zero_grad()
                self.optimizer_C.zero_grad()
                #total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()    
                total_hit += torch.sum(torch.argmax(class_logits, dim=1) == torch.argmax(source_label, dim=1)).item()
                total_num += source_data.shape[0]

                #return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num
                train_D_loss= running_D_loss / (i+1)
                train_F_loss= running_F_loss / (i+1)
                train_acc= total_hit / total_num
               
                
                if train_D_loss+train_F_loss< min_loss:
                    min_loss= train_D_loss+train_F_loss
                    torch.save(self.feature_extractor.state_dict(), f"../weight/P{self.patient}_extractor_model.bin")
                    torch.save(self.label_predictor.state_dict(), f"../weight/P{self.patient}_predictor_model.bin")
            if epoch%self.interval==0:
                
                # for name, param in self.label_predictor.named_parameters():
                #     if param.grad is not None:
                #         print(f'Parameter: {name}, Gradient norm: {param.grad.norm()}')
                # for name, param in self.label_predictor.named_parameters():
                #     if param.grad is not None:
                #         print(f'Parameter: {name}, Gradient norm: {param.grad.norm()}')
                print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))


    def evaluate(self, dataset, test_patients):
        # Evaluation code here
        pred_labels = []
        true_labels = []
        # Loading model state dicts and setting to evaluation mode
        self.label_predictor.load_state_dict(torch.load(f"../weight/P{self.patient}_predictor_model.bin"))
        self.feature_extractor.load_state_dict(torch.load(f"../weight/P{self.patient}_extractor_model.bin"))
        self.label_predictor.eval()
        self.feature_extractor.eval()
        for i, (test_data, test_label) in enumerate(dataset):
            test_data = test_data.to(self.device)
            class_logits = self.label_predictor(self.feature_extractor(test_data))
            pred_label = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
            pred_labels.append(pred_label)
            true_labels.append(test_label.cpu().numpy())

        pred_labels= np.concatenate(pred_labels)
        true_labels= np.concatenate(true_labels)
        #print('pred',pred_labels)
        #print('true',true_labels)
        
        
        if len(test_patients)>1:
            self.acc1, self.sensitivity1, self.specificity1, self.acc2, self.sensitivity2, self.specificity2= metrics_individual(true_labels ,pred_labels)
        else:
            self.acc1, self.sensitivity1, self.specificity1= metrics(true_labels ,pred_labels)
        return self
    def only_prediction(self, dataset):
        # Evaluation code here
        pred_labels = []
        true_labels = []
        # Loading model state dicts and setting to evaluation mode
        self.label_predictor.load_state_dict(torch.load(f"../weight/P{self.patient}_predictor_model.bin"))
        self.feature_extractor.load_state_dict(torch.load(f"../weight/P{self.patient}_extractor_model.bin"))
        self.label_predictor.eval()
        self.feature_extractor.eval()
        for i, (test_data, test_label) in enumerate(dataset):
            test_data = test_data.to(self.device)
            class_logits = self.label_predictor(self.feature_extractor(test_data))
            pred_label = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
            pred_labels.append(pred_label)
            true_labels.append(test_label.cpu().numpy())
    
        
        pred_labels= np.concatenate(pred_labels)
        true_labels= np.concatenate(true_labels)
        

        return pred_labels

class LSTMTrainer:
    def __init__(self,patient,  model, train_dataloader, val_dataloader, test_dataloader, \
                   criterion, optimizer, device, max_epochs, interval):
        self.patient= patient
        self.model= model
        self.train_dataloader= train_dataloader
        self.val_dataloader= val_dataloader
        self.test_dataloader= test_dataloader
        self.criterion= criterion
        self.optimizer= optimizer
        self.device= device
        self.max_epochs= max_epochs
        self.interval = interval

 

    def evaluate(self):
        pred_labels, true_labels= [], []
        self.model.load_state_dict(torch.load(f'P{self.patient}_weight.pth'))
        self.model.eval()

        with torch.no_grad():
            for data, target in self.test_dataloader:
                target =F.one_hot( target , num_classes=2)
                target = target .type(torch.FloatTensor)
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                target= target.argmax(dim=1)
                pred_label = torch.argmax(output, dim=1).cpu().detach().numpy()
                pred_labels.append(pred_label)
                true_labels.append(target.cpu().numpy())
        
        pred_labels= np.concatenate(pred_labels)
        true_labels= np.concatenate(true_labels)
        print('pred',pred_labels)
        print('true',true_labels)
        self.acc, self.sensitivity, self.specificity =metrics(true_labels, pred_labels)
        return self
    def train(self):
       
        self.model.to(self.device)
        min_loss=1e10
        history_train = {'loss': [], 'acc': []}
        history_val = {'loss': [], 'acc': []}

        
        for epoch in range(self.max_epochs):
      
            num_correct = 0
            val_num_correct=0
            train_len, val_len=0,0
            train_loss, val_loss=0.0, 0.0
            self.model.train() 
            
            for i,( data, target )in enumerate( self.train_dataloader):
                target =F.one_hot( target , num_classes=2)
                target = target .type(torch.FloatTensor)
                data, target = data.to(self.device), target.to(self.device)
                output,_ = self.model(data) 
                loss = self.criterion(output, target) 
                train_loss+=loss
                self.optimizer.zero_grad()
                loss.backward()  
                self.optimizer.step()
                y_pred = output.argmax(dim=1)
                target= target.argmax(dim=1)
                num_correct += (y_pred == target).sum().item()
                train_len+= len(target)

            train_accuracy = float(num_correct) / train_len * 100
            train_loss= train_loss/(i+1)#train_len
            history_train['loss'].append(train_loss)
            history_train['acc'].append(train_accuracy)

            self.model.eval()
            with torch.no_grad():
                for i,( data, target )in enumerate( self.val_dataloader):
                    target =F.one_hot( target , num_classes=2)
                    target = target .type(torch.FloatTensor)
                    data, target = data.to(self.device), target.to(self.device)
                    output,_ = self.model(data) 
                    loss = self.criterion(output, target) 
                    val_loss+=loss
                    y_pred = output.argmax(dim=1)
                    target= target.argmax(dim=1)
                    val_num_correct += (y_pred == target).sum().item()
                    val_len+= len(target)

            val_accuracy = float(val_num_correct) / val_len * 100
            val_loss= val_loss/(i+1)#train_len
            history_val['loss'].append(val_loss)
            history_val['acc'].append(val_accuracy)


            if val_loss < min_loss:
                min_loss= val_loss
            
                torch.save(self.model.state_dict(),f'P{self.patient}_weight.pth')

            if (epoch+1)%self.interval ==0:
          
                print(f'[Epoch {epoch + 1}/{self.max_epochs}]'
                    f" loss: {history_train['loss'][-1]:.4f}, acc: {history_train['acc'][-1]:2.2f}%"
                    f" loss: {history_val['loss'][-1]:.4f}, acc: {history_val['acc'][-1]:2.2f}%")
    
    
