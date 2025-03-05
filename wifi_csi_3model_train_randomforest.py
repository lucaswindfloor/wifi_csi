import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Configuration parameters
MODEL_PATH = "rf_model.joblib"  # Random forest model save path
DATA_PATH = "processed_data/all_processed_data.pkl"
BASELINE_PATH = "baseline_features.npz"  # Baseline feature save path
TEST_SIZE = 0.2  # Test set proportion
RANDOM_SEED = 42  # Random seed
N_ESTIMATORS = 50  # Number of trees in random forest
MAX_DEPTH = 8  # Maximum tree depth

# Create directory if needed
model_dir = os.path.dirname(MODEL_PATH)
if model_dir:
    os.makedirs(model_dir, exist_ok=True)


# Load data function
def load_data(data_path):
    """Load CSI data from pickle file"""
    print(f"Loading data: {data_path}")
    try:
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)

        features = []
        labels = []
        feature_names = None

        for result in all_data:
            current_features = result['features']
            current_label = result['label']

            # Save feature names
            if feature_names is None and 'feature_names' in result:
                feature_names = result['feature_names']

            # Add features and labels from each window
            for feature in current_features:
                features.append(feature)
                labels.append(current_label)

        return np.array(features), np.array(labels), all_data, feature_names
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


# Train random forest model
def train_random_forest(X_train, y_train, X_val, y_val, feature_names=None):
    """Train and evaluate random forest model"""
    print("Training random forest model...")

    # Create model
    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,  # Use all CPU cores
        class_weight='balanced'  # Handle potential class imbalance
    )

    # Train model
    rf_model.fit(X_train, y_train)

    # Evaluate model
    train_acc = rf_model.score(X_train, y_train)
    val_acc = rf_model.score(X_val, y_val)

    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

    # Predict validation set
    y_pred = rf_model.predict(X_val)

    # Print detailed evaluation report
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, target_names=['No Person', 'Person']))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("\nConfusion matrix:")
    print(cm)

    # Visualize feature importance
    if feature_names is not None:
        visualize_feature_importance(rf_model, feature_names)

    # Save model
    joblib.dump(rf_model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    return rf_model, train_acc, val_acc


# Optional: Hyperparameter optimization
def optimize_hyperparams(X_train, y_train):
    """Find optimal hyperparameters using grid search"""
    print("\nPerforming hyperparameter optimization...")

    param_grid = {
        'n_estimators': [30, 50, 100],
        'max_depth': [6, 8, 10, None],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 4]
    }

    base_model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# Visualize feature importance with English labels only
def visualize_feature_importance(model, feature_names, top_n=15):
    """Visualize the importance of features in the random forest model"""
    print("Visualizing feature importance...")

    if not hasattr(model, 'feature_importances_'):
        print("Model doesn't have feature_importances_ attribute")
        return

    # Get feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Select top features
    n_features = min(top_n, len(feature_names))

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance (Random Forest)', fontsize=14)
    plt.bar(range(n_features), importances[indices[:n_features]], align='center')

    # Format feature names to ensure they're in English
    formatted_names = []
    for i in indices[:n_features]:
        name = feature_names[i]
        # Replace any potential non-ASCII characters
        formatted_name = ''.join(c if ord(c) < 128 else '_' for c in name)
        formatted_names.append(formatted_name)

    plt.xticks(range(n_features), formatted_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.ylabel('Importance Score', fontsize=12)
    plt.xlabel('Feature', fontsize=12)

    # Save image
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("Feature importance plot saved as 'feature_importance.png'")
    plt.show()


# Extract baseline features from "no person" samples
def extract_baseline_features(all_data, feature_names, output_path=BASELINE_PATH):
    """Extract baseline features from no-person state data"""
    print(f"\nExtracting baseline features...")

    try:
        # Collect all "no person" sample features
        no_person_features = []

        for result in all_data:
            if result['label'] == 0:  # 0 means no person
                no_person_features.extend(result['features'])

        no_person_features = np.array(no_person_features)
        print(f"Collected {len(no_person_features)} no-person state feature samples")

        if len(no_person_features) == 0:
            print("Error: No no-person state samples found")
            return False

        # Calculate feature mean and standard deviation
        baseline_mean = np.mean(no_person_features, axis=0)
        baseline_std = np.std(no_person_features, axis=0)

        # Calculate feature range
        baseline_min = np.min(no_person_features, axis=0)
        baseline_max = np.max(no_person_features, axis=0)

        # Save some original feature samples for later use
        sample_count = min(20, len(no_person_features))
        sample_features = no_person_features[:sample_count]

        # Format feature names to ensure they're ASCII-safe
        safe_feature_names = []
        for name in feature_names:
            safe_name = ''.join(c if ord(c) < 128 else '_' for c in name)
            safe_feature_names.append(safe_name)

        # Save baseline features
        np.savez(output_path,
                 mean=baseline_mean,
                 std=baseline_std,
                 min=baseline_min,
                 max=baseline_max,
                 feature_names=safe_feature_names,
                 samples=sample_features,
                 sample_count=len(no_person_features),
                 timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        print(f"Baseline features saved to: {output_path}")
        print(f"Feature range: [{np.min(baseline_mean):.3f}, {np.max(baseline_mean):.3f}]")
        print(f"Std dev range: [{np.min(baseline_std):.3f}, {np.max(baseline_std):.3f}]")
        return True

    except Exception as e:
        print(f"Failed to extract baseline features: {e}")
        import traceback
        traceback.print_exc()
        return False


# Main function
def main():
    print("CSI Human Presence Detection - Random Forest Model Training")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    features, labels, all_data, feature_names = load_data(DATA_PATH)
    print(f"Loaded {len(features)} samples with {features.shape[1]} features each")
    print(f"Class distribution: No Person={sum(labels == 0)}, Person={sum(labels == 1)}")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Train random forest model
    rf_model, train_acc, val_acc = train_random_forest(X_train, y_train, X_val, y_val, feature_names)

    # Optional: Perform hyperparameter optimization
    # best_model = optimize_hyperparams(X_train, y_train)
    # joblib.dump(best_model, "rf_model_optimized.joblib")
    # print("Optimized model saved as 'rf_model_optimized.joblib'")

    # Extract and save baseline features
    extract_baseline_features(all_data, feature_names)

    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Final model saved to: {MODEL_PATH}")
    print(f"Baseline features saved to: {BASELINE_PATH}")

    # Show final performance summary
    print("\nPerformance Summary:")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print("Random forest model training complete!")


if __name__ == "__main__":
    main()