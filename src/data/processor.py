# Authored by Suhrit Ghimire
import pandas as pd
import numpy as np
import re
import spacy
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import Dataset, DatasetDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HateSpeechDataProcessor:
    """
    Consolidated data engineering pipeline for Hate Speech Detection comparative experiments.
    Handles text normalization, SpaCy linguistic cleaning, SMOTE oversampling for sequential models,
    and Hugging Face Dataset formatting for Transformers (BERT, RoBERTa, DistilBERT).
    """
    
    def __init__(self, data_path, vocab_size=10000, max_len=30):
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.max_len = max_len
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            logger.warning("SpaCy model not found, falling back to regex tokenization.")
            self.nlp = None
        self.df = None
        
    def load_data(self):
        logger.info(f"Loading dataset from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        # Drop non-essential metadata columns if present
        cols_to_drop = ["Unnamed: 0", "count", "hate_speech", "offensive_language", "neither"]
        self.df = self.df.drop(columns=[col for col in cols_to_drop if col in self.df.columns])
        self.df["class"] = self.df["class"].astype(int)
        logger.info(f"Loaded {len(self.df)} samples. Class distribution:\n{self.df['class'].value_counts()}")
        return self.df

    @staticmethod
    def clean_text(text):
        """Normalizes tweet text by stripping handles, URLs, and non-alphanumeric chars."""
        if not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Remove @user mentions
        text = re.sub(r"@[A-Za-z0-9_]+", "", text)
        # Remove RT tags
        text = re.sub(r"\bRT\b", "", text)
        # Keep letters and basic punctuation
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def lemmatize_text(self, text):
        """Performs lemmatization and stopword removal via SpaCy."""
        if self.nlp is None:
            return text.lower()
        doc = self.nlp(text)
        return " ".join([token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha])

    def prepare_classical_data(self, test_size=0.2, random_state=42):
        """Prepares raw cleaned text splits for TF-IDF + Classical Classifiers."""
        logger.info("Preparing data for TF-IDF + SVM baseline...")
        self.df["cleaned"] = self.df["tweet"].apply(self.clean_text)
        return train_test_split(
            self.df["cleaned"].values,
            self.df["class"].values,
            test_size=test_size,
            stratify=self.df["class"].values,
            random_state=random_state
        )

    def prepare_lstm_data(self, use_attention=False, test_size=0.2, random_state=42):
        """Prepares integer-encoded padded sequences with optional SMOTE for LSTM/BiLSTM."""
        logger.info(f"Preparing data for {'BiLSTM + Attention' if use_attention else 'LSTM'} pipeline...")
        self.df["cleaned"] = self.df["tweet"].apply(self.clean_text)
        self.df["processed"] = self.df["cleaned"].apply(self.lemmatize_text)
        
        encoded = [one_hot(text, self.vocab_size) for text in self.df["processed"]]
        padded = pad_sequences(encoded, padding="post", maxlen=self.max_len)
        
        X = np.array(padded)
        y = np.array(self.df["class"])
        
        # Apply SMOTE oversampling to balance minority classes
        min_samples = self.df["class"].value_counts().min()
        if min_samples >= 6:
            smote = SMOTE(sampling_strategy="auto", random_state=random_state)
            X_res, y_res = smote.fit_resample(X, y)
        else:
            X_res, y_res = X, y
            
        return train_test_split(
            X_res, y_res,
            test_size=test_size,
            stratify=y_res,
            random_state=random_state
        )

    def prepare_transformer_data(self, test_size=0.2, valid_size=0.1, random_state=42):
        """Prepares stratified Hugging Face DatasetDict for BERT, RoBERTa, and DistilBERT."""
        logger.info("Preparing stratified splits for Transformer models...")
        self.df["tweet_cleaned"] = self.df["tweet"].apply(self.clean_text)
        
        # Stratified train / test split
        train_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            stratify=self.df["class"],
            random_state=random_state
        )
        
        # Stratified train / val split
        train_df, val_df = train_test_split(
            train_df,
            test_size=valid_size,
            stratify=train_df["class"],
            random_state=random_state
        )
        
        return DatasetDict({
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "valid": Dataset.from_pandas(val_df.reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df.reset_index(drop=True))
        })
