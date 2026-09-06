# Authored by Suhrit Ghimire
import argparse
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from src.data.processor import HateSpeechDataProcessor
from src.models.architectures import (
    TransformerClassifier,
    BiLSTMWithAttentionClassifier,
    LSTMBasedClassifier,
    TFIDF_SVMClassifier
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "macro_f1": f1, "macro_precision": precision, "macro_recall": recall}

def main():
    parser = argparse.ArgumentParser(description="Hate Speech Detection: Comparative Transformer & Neural Suite")
    parser.add_argument(
        "--model",
        type=str,
        choices=["bert", "roberta", "distilbert", "bilstm_attention", "bilstm", "lstm", "svm_tfidf"],
        default="bert",
        help="Model architecture to train and evaluate"
    )
    parser.add_argument("--data", type=str, default="src/data/labeled_data.csv", help="Path to labeled data CSV")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze transformer backbone encoder layers")
    parser.add_argument("--sample_mode", action="store_true", help="Run quick dry-run test on a 200-sample subset")
    
    args = parser.parse_args()
    
    processor = HateSpeechDataProcessor(args.data)
    processor.load_data()
    
    if args.sample_mode:
        processor.df = processor.df.sample(min(200, len(processor.df)), random_state=42)
        logger.info("Running in smoke-test sample mode with 200 samples.")

    # ─────────────────────────────────────────────────────────────
    # Classical Baseline: TF-IDF + SVM
    # ─────────────────────────────────────────────────────────────
    if args.model == "svm_tfidf":
        X_train, X_test, y_train, y_test = processor.prepare_classical_data()
        classifier = TFIDF_SVMClassifier()
        classifier.fit(X_train, y_train)
        preds = classifier.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro")
        logger.info(f"=== TF-IDF + SVM Results ===")
        logger.info(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Macro Precision: {p:.4f} | Macro Recall: {r:.4f}")
        print(classification_report(y_test, preds, target_names=["Hate Speech", "Offensive", "Neither"]))

    # ─────────────────────────────────────────────────────────────
    # Sequential Baselines: LSTM / BiLSTM / BiLSTM + Attention
    # ─────────────────────────────────────────────────────────────
    elif args.model in ["lstm", "bilstm", "bilstm_attention"]:
        use_att = (args.model == "bilstm_attention")
        X_train, X_test, y_train, y_test = processor.prepare_lstm_data(use_attention=use_att)
        
        if use_att:
            model = BiLSTMWithAttentionClassifier.build_model(vocab_size=processor.vocab_size)
        else:
            model = LSTMBasedClassifier.build_model(
                vocab_size=processor.vocab_size,
                bidirectional=(args.model == "bilstm")
            )
            
        model.fit(
            X_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_test, y_test)
        )
        
        preds_prob = model.predict(X_test)
        preds = np.argmax(preds_prob, axis=1)
        acc = accuracy_score(y_test, preds)
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro")
        logger.info(f"=== {args.model.upper()} Results ===")
        logger.info(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Macro Precision: {p:.4f} | Macro Recall: {r:.4f}")
        print(classification_report(y_test, preds, target_names=["Hate Speech", "Offensive", "Neither"]))

    # ─────────────────────────────────────────────────────────────
    # Transformers: BERT / RoBERTa / DistilBERT
    # ─────────────────────────────────────────────────────────────
    elif args.model in ["bert", "roberta", "distilbert"]:
        dataset = processor.prepare_transformer_data()
        classifier = TransformerClassifier(model_key=args.model, num_labels=3)
        
        if args.freeze_backbone:
            classifier.freeze_backbone()
            
        def tokenize_fn(examples):
            return classifier.tokenizer(
                examples["tweet_cleaned"],
                padding="max_length",
                truncation=True,
                max_length=128
            )
            
        tokenized_ds = dataset.map(tokenize_fn, batched=True)
        tokenized_ds = tokenized_ds.rename_column("class", "labels")
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        from transformers import TrainingArguments
        training_args = TrainingArguments(
            output_dir=f"./experiments_{args.model}",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=2e-5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir="./logs",
            report_to="none"
        )
        
        trainer = classifier.get_trainer(
            train_ds=tokenized_ds["train"],
            eval_ds=tokenized_ds["valid"],
            training_args=training_args,
            compute_metrics=compute_metrics
        )
        
        logger.info(f"Starting fine-tuning for {args.model.upper()}...")
        trainer.train()
        eval_results = trainer.evaluate(tokenized_ds["test"])
        logger.info(f"=== {args.model.upper()} Test Evaluation Results ===")
        for k, v in eval_results.items():
            logger.info(f"  {k}: {v}")

if __name__ == "__main__":
    main()
