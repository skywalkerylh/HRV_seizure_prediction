import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from data_preprocessing import IO, LSTMDataset, preprocessing_pipeline
from model import (
    FeatureExtractor, 
    DomainClassifier, 
    LabelPredictor,
    FocalLoss, 
    LSTM, 
    generate_LOSO_train_test_subjects, 
    initialize_model
)
from train import DomainAdaptationTrainer, LSTMTrainer
from utils import record_metrics, gen_pred_info, plot_tsne, visualize, set_seed

@dataclass
class ModelConfig:
    """Configuration class for model hyperparameters"""
    hidden_size: int = 512
    num_layers: int = 2
    bidirectional: bool = True
    domain_classifier_hidden: int = 512
    learning_rate: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 30
    sequence_length: int = 5
    interval: int = 10

@dataclass
class DataConfig:
    """Configuration class for data preprocessing parameters"""
    normalization: bool = True
    downsampling: bool = False
    select_random_rows: bool = False
    features: List[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = [
                'RRMean', 'SDNN', 'RMSSD', 'pNN50', 'TOTAL_POWER',
                'VLF_POWER', 'LF_POWER', 'HF_POWER', 'LF_TO_HF', 'SampEn'
            ]

class DataProcessor:
    """Handles data loading and preprocessing operations"""
    def __init__(self, data_config: DataConfig):
        self.config = data_config
        self.seizure_nums = [10, 12, 15, 17, 18, 36]
        self.seizure_nums_nolabel = [1, 2, 3, 4, 13, 20, 25, 27, 28, 33]
        self.arrhythmia = [1, 2, 3, 4, 8, 9, 17, 32]

    def load_and_preprocess_data(
        self, 
        patient: int, 
        folder: str,
        idx: int
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and preprocess data for a single patient"""
        path = f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/feature_P{patient}.csv"
        
        # Handle special cases for different patient types
        if patient in self.arrhythmia:
            folder = '0909_overlap_epochingbyPoints'
            path = f"G:/我的雲端硬碟/thesis/code/ECG/features/{folder}/feature_P{patient}.csv"
            
        dataset = IO.read_raw_csv(path, patient, folder, idx)
        dataset.data['patient'] = idx
        
        processed_dataset = preprocessing_pipeline(
            dataset,
            self.seizure_nums,
            self.seizure_nums_nolabel,
            patient,
            self.config.features,
            downsampling=self.config.downsampling,
            normalization=self.config.normalization,
            select_random_rows=self.config.select_random_rows,
            ds_freq=2
        )
        
        self._handle_missing_values(processed_dataset)
        return processed_dataset

    def _handle_missing_values(self, dataset) -> None:
        """Handle missing values in the dataset"""
        if np.any(np.isnan(dataset.data)):
            nan_rows = dataset.data[dataset.data.isnull().any(axis=1)]
            print("Imputing missing values...")
            dataset.data = dataset.data.ffill()

class ExperimentRunner:
    """Manages experiment execution"""
    def __init__(
        self, 
        model_config: ModelConfig,
        data_config: DataConfig,
        data_processor: DataProcessor
    ):
        self.model_config = model_config
        self.data_config = data_config
        self.data_processor = data_processor
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def run_internal_validation(
        self,
        train_patients: List[int],
        val_patients: List[int],
        test_patients: List[int]
    ) -> Dict:
        """Run internal validation experiment"""
        # Prepare datasets
        train_data = self._prepare_training_data(train_patients)
        test_data = self._prepare_test_data(test_patients)
        
        # Initialize model and trainer
        model_components = self._initialize_model_components()
        trainer = self._setup_trainer(model_components, train_data, test_data)
        
        # Train and evaluate
        trainer.train()
        results = trainer.evaluate(test_data, test_patients)
        
        return self._format_results(results)

    def _prepare_training_data(self, patients: List[int]) -> DataLoader:
        """Prepare training data"""
        dataset = None
        for idx, patient in enumerate(patients):
            current_dataset = self.data_processor.load_and_preprocess_data(
                patient, '0909_overlap_epochingbyPoints', idx
            )
            dataset = current_dataset if dataset is None else dataset.append_dataset(current_dataset)

        if self.data_config.normalization:
            dataset = self._apply_smote(dataset)
            
        return self._create_dataloader(dataset)

    def _apply_smote(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE for handling class imbalance"""
        smote = SMOTE()
        return smote.fit_resample(dataset.data, dataset.label)

    def _create_dataloader(self, dataset) -> DataLoader:
        """Create DataLoader from dataset"""
        dataset = LSTMDataset(
            dataset.data.to_numpy(),
            dataset.label.to_numpy(),
            self.model_config.sequence_length,
            self.data_config.normalization
        )
        return DataLoader(dataset, batch_size=self.model_config.batch_size, shuffle=False)

    def _initialize_model_components(self) -> Tuple:
        """Initialize model components"""
        return initialize_model(
            name='dann',
            hidden_size=self.model_config.hidden_size,
            num_layers=self.model_config.num_layers,
            bidirectional=self.model_config.bidirectional,
            device=self.device,
            features=self.data_config.features,
            lr=self.model_config.learning_rate,
            domain_classifier_hidden=self.model_config.domain_classifier_hidden
        )

    def _setup_trainer(
        self,
        model_components: Tuple,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader
    ) -> DomainAdaptationTrainer:
        """Setup the trainer"""
        (
            feature_extractor,
            label_predictor,
            domain_classifier,
            _,
            class_criterion,
            domain_criterion,
            optimizer_C,
            optimizer_D,
            optimizer_F
        ) = model_components

        return DomainAdaptationTrainer(
            feature_extractor,
            label_predictor,
            domain_classifier,
            optimizer_F,
            optimizer_C,
            optimizer_D,
            train_dataloader,
            test_dataloader,
            self.model_config.max_epochs,
            self.model_config.interval,
            self.device,
            domain_criterion,
            class_criterion
        )

def main():
    """Main entry point"""
    # Initialize configurations
    model_config = ModelConfig()
    data_config = DataConfig()
    data_processor = DataProcessor(data_config)
    
    # Setup experiment
    experiment = ExperimentRunner(model_config, data_config, data_processor)
    
    # Run experiment
    train_patients = [10, 12]
    val_patients = [11, 15]
    test_patients = [18, 36, 24, 26, 21]
    
    results = experiment.run_internal_validation(train_patients, val_patients, test_patients)
    print("Experiment Results:", results)

if __name__ == "__main__":
    main()
