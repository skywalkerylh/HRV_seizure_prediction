import torch
import numpy as np
from utils import metrics, metrics_individual

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
            # D loss: Domain Classifier的loss
            # F loss: Feature Extrator & Label Predictor的loss
            # total_hit: 計算目前對了幾筆 total_num: 目前經過了幾筆
            running_D_loss, running_F_loss, min_loss = 0.0, 0.0, 1e10
            total_hit, total_num = 0.0, 0.0

            for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(self.dataloader_source, self.dataloader_target)):

                source_data = source_data.cuda()
                source_label = source_label.cuda()
                target_data = target_data.cuda()
                
                # 我們把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
                mixed_data = torch.cat([source_data, target_data], dim=0)
                domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
                # 設定source data的label為1
                domain_label[:source_data.shape[0]] = 1

                # Step 1 : 訓練Domain Classifier
                feature_extractor = self.feature_extractor(mixed_data)
                # 因為我們在Step 1不需要訓練Feature Extractor，所以把feature detach避免loss backprop上去。
                domain_logits = self.domain_classifier(feature_extractor.detach())
                loss = self.domain_criterion(domain_logits, domain_label)
                running_D_loss+= loss.item()
                loss.backward()
                self.optimizer_D.step()

                # Step 2 : 訓練Feature Extractor和Domain Classifier
                class_logits = self.label_predictor(feature_extractor[:source_data.shape[0]])
                domain_logits = self.domain_classifier(feature_extractor)
                # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。
                loss = self.class_criterion(class_logits, source_label) - lamb * self.domain_criterion(domain_logits, domain_label)
                running_F_loss+= loss.item()
                loss.backward()
                self.optimizer_F.step()
                self.optimizer_C.step()

                self.optimizer_D.zero_grad()
                self.optimizer_F.zero_grad()
                self.optimizer_C.zero_grad()

                total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
                total_num += source_data.shape[0]

                #return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num
                train_D_loss= running_D_loss / (i+1)
                train_F_loss= running_F_loss / (i+1)
                train_acc= total_hit / total_num
                # rec6rd
                
                if train_D_loss+train_F_loss< min_loss:
                    min_loss= train_D_loss+train_F_loss
                    torch.save(self.feature_extractor.state_dict(), f"../weight/P{self.patient}_extractor_model.bin")
                    torch.save(self.label_predictor.state_dict(), f"../weight/P{self.patient}_predictor_model.bin")
            if epoch%self.interval==0:
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
        print('pred',pred_labels)
        print('true',true_labels)
        
        metrics(true_labels ,pred_labels)
        if len(test_patients)>1:
            metrics_individual(true_labels ,pred_labels)