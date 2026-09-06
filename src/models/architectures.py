# Authored by Suhrit Ghimire
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

class TransformerClassifier:
    \"\"\"
    Unified Transformer classifier supporting BERT, RoBERTa, and DistilBERT
    with backbone freezing and custom classification head capabilities.
    \"\"\"
    
    MODEL_MAP = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "distilbert": "distilbert-base-uncased"
    }

    def __init__(self, model_key="bert", num_labels=3):
        self.model_key = model_key.lower()
        self.model_name = self.MODEL_MAP.get(self.model_key, model_key)
        self.num_labels = num_labels
        logger.info(f"Loading transformer checkpoint: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )

    def freeze_backbone(self):
        \"\"\"Freezes the transformer encoder layers, leaving only classification head trainable.\"\"\"
        logger.info(f"Freezing backbone for {self.model_name}...")
        if hasattr(self.model, "bert"):
            for param in self.model.bert.parameters():
                param.requires_grad = False
        elif hasattr(self.model, "roberta"):
            for param in self.model.roberta.parameters():
                param.requires_grad = False
        elif hasattr(self.model, "distilbert"):
            for param in self.model.distilbert.parameters():
                param.requires_grad = False
        return self

    def get_trainer(self, train_ds, eval_ds, training_args, compute_metrics):
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )


class AttentionLayer(layers.Layer):
    \"\"\"
    Custom Bahdanau / Context Attention Layer for BiLSTM sequence outputs.
    Calculates attention weights across all timesteps to generate a context vector.
    \"\"\"
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (batch_size, time_steps, features)
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        a = keras.backend.softmax(e, axis=1)
        output = x * a
        return keras.backend.sum(output, axis=1)


class BiLSTMWithAttentionClassifier:
    \"\"\"
    Bidirectional LSTM with Self-Attention mechanism for sequence toxicity detection.
    \"\"\"
    @staticmethod
    def build_model(vocab_size, embedding_dim=100, max_len=30, num_classes=3):
        logger.info("Building BiLSTM + Attention baseline model...")
        inputs = layers.Input(shape=(max_len,))
        emb = layers.Embedding(vocab_size, embedding_dim, input_length=max_len)(inputs)
        bilstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(emb)
        att = AttentionLayer()(bilstm)
        dense = layers.Dense(64, activation="relu")(att)
        drop = layers.Dropout(0.3)(dense)
        outputs = layers.Dense(num_classes, activation="softmax")(drop)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model


class LSTMBasedClassifier:
    \"\"\"Standard Sequential LSTM model.\"\"\"
    @staticmethod
    def build_model(vocab_size, embedding_dim=50, max_len=20, bidirectional=False):
        logger.info(f"Building {\"Bi-Directional \" if bidirectional else \"\"}LSTM model...")
        model = keras.Sequential()
        model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_len))
        
        if bidirectional:
            model.add(layers.Bidirectional(layers.LSTM(100, return_sequences=True)))
            model.add(layers.Bidirectional(layers.LSTM(50)))
        else:
            model.add(layers.LSTM(100, return_sequences=True))
            model.add(layers.LSTM(50))
            
        model.add(layers.Dense(3, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model


class TFIDF_SVMClassifier:
    \"\"\"
    Classical ML Baseline: TF-IDF (Unigrams + Bigrams) with Linear Support Vector Classifier.
    \"\"\"
    def __init__(self, max_features=10000):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, sublinear_tf=True)),
            ("svm", LinearSVC(C=1.0, class_weight="balanced", random_state=42))
        ])

    def fit(self, X_train, y_train):
        logger.info("Training TF-IDF + SVM baseline...")
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.pipeline.predict(X_test)
