
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import pickle
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config import DATA_CONFIG, CLASSIFIER_CONFIG
from utils import get_logger

logger = get_logger(__name__)

def train_classifier(X: np.ndarray, y: list) -> RandomForestClassifier:
    """Train RandomForestClassifier from CLASSIFIER_CONFIG parameters."""
    n_estimators = CLASSIFIER_CONFIG.get("n_estimators", 100)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X, y)
    return clf

def evaluate_classifier(clf: RandomForestClassifier, X_test: np.ndarray, y_test: list) -> dict:
    """Evaluate the classifier and return per-class metrics."""
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    
    print("\n--- Attack Type Classifier Performance ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    return report

def main():
    """Load data, train and evaluate model, and save."""
    gtae_path = DATA_CONFIG["gtae_results_path"]
    if not os.path.exists(gtae_path):
        logger.error(f"GTAE results not found at {gtae_path}")
        return

    with open(gtae_path, "rb") as f:
        results = pickle.load(f)

    # Note: Using .numpy() if it's torch tensor, otherwise just assume it's array
    embeddings = results["embeddings"]
    if hasattr(embeddings, "numpy"):
        embeddings = embeddings.numpy()
        
    edge_labels = results["edge_labels"]
    if hasattr(edge_labels, "numpy"):
        edge_labels = edge_labels.numpy()
        
    attack_types = results["attack_types"]

    # Filter to attack-labeled flows only (edge label 1)
    attack_mask = edge_labels == 1
    X = embeddings[attack_mask]
    y = [attack_types[i] for i in range(len(attack_types)) if attack_mask[i]]

    unique_classes = set(y)
    if len(unique_classes) < 2:
        logger.warning(f"Fewer than 2 unique classes found ({unique_classes}). Cannot train classifier. Exiting gracefully.")
        print(f"Not enough attack type variety to train a classifier (found: {unique_classes}). Exiting.")
        return

    # Stratified 75/25 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logger.info("Training Attack Type Classifier...")
    clf = train_classifier(X_train, y_train)

    logger.info("Evaluating Attack Type Classifier...")
    evaluate_classifier(clf, X_test, y_test)

    model_path = CLASSIFIER_CONFIG["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    logger.info(f"Saved trained classifier to {model_path}")
    print(f"Saved trained classifier to {model_path}")

if __name__ == "__main__":
    main()
