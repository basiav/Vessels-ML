import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from skimage import feature, filters, measure
import warnings
import traceback

warnings.filterwarnings("ignore")


class LesionFeatureExtractor:
    def __init__(self, image_size=(100, 100)):
        self.image_size = image_size

    def extract_intensity_features(self, image):
        features = {}

        # Basic statistics
        features["mean_intensity"] = np.mean(image)
        features["std_intensity"] = np.std(image)
        features["median_intensity"] = np.median(image)
        features["min_intensity"] = np.min(image)
        features["max_intensity"] = np.max(image)
        features["intensity_range"] = (
            features["max_intensity"] - features["min_intensity"]
        )

        # Percentiles
        features["p25_intensity"] = np.percentile(image, 25)
        features["p75_intensity"] = np.percentile(image, 75)
        features["iqr_intensity"] = (
            features["p75_intensity"] - features["p25_intensity"]
        )

        # Skewness and kurtosis approximation
        mean_val = features["mean_intensity"]
        std_val = features["std_intensity"]
        if std_val > 0:
            normalized = (image - mean_val) / std_val
            features["skewness"] = np.mean(normalized**3)
            features["kurtosis"] = np.mean(normalized**4) - 3
        else:
            features["skewness"] = 0
            features["kurtosis"] = 0

        return features

    def extract_texture_features(self, image):
        features = {}

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image_uint8 = (
                (image - image.min()) / (image.max() - image.min()) * 255
            ).astype(np.uint8)
        else:
            image_uint8 = image

        # Local Binary Patterns
        lbp = feature.local_binary_pattern(image_uint8, P=8, R=1, method="uniform")
        features["lbp_mean"] = np.mean(lbp)
        features["lbp_std"] = np.std(lbp)

        # Gradient features
        grad_x = filters.sobel_h(image)
        grad_y = filters.sobel_v(image)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features["gradient_mean"] = np.mean(gradient_magnitude)
        features["gradient_std"] = np.std(gradient_magnitude)
        features["gradient_max"] = np.max(gradient_magnitude)

        return features

    def extract_morphological_features(self, image):
        features = {}

        # Threshold image to create binary mask
        threshold = filters.threshold_otsu(image)
        binary = image > threshold

        # Basic morphological properties
        features["object_area"] = np.sum(binary)
        features["object_perimeter"] = measure.perimeter(binary)

        if features["object_area"] > 0:
            features["circularity"] = (
                4
                * np.pi
                * features["object_area"]
                / (features["object_perimeter"] ** 2)
            )
        else:
            features["circularity"] = 0

        # Moments
        moments = measure.moments(binary.astype(float))
        if moments[0, 0] > 0:
            centroid_y = moments[1, 0] / moments[0, 0]
            centroid_x = moments[0, 1] / moments[0, 0]
            features["centroid_x"] = centroid_x / image.shape[1]
            features["centroid_y"] = centroid_y / image.shape[0]
        else:
            features["centroid_x"] = 0.5
            features["centroid_y"] = 0.5

        # Compactness
        features["compactness"] = features["object_area"] / (
            self.image_size[0] * self.image_size[1]
        )

        return features

    def extract_spatial_features(self, image):
        features = {}

        h, w = image.shape
        quadrants = [
            image[: h // 2, : w // 2],  # TL
            image[: h // 2, w // 2 :],  # TR
            image[h // 2 :, : w // 2],  # BL
            image[h // 2 :, w // 2 :],  # BR
        ]

        for i, quad in enumerate(quadrants):
            features[f"quad_{i}_mean"] = np.mean(quad)
            features[f"quad_{i}_std"] = np.std(quad)

        # Center vs periphery
        center_mask = np.zeros_like(image, dtype=bool)
        cy, cx = image.shape[0] // 2, image.shape[1] // 2
        radius = min(image.shape) // 4
        y, x = np.ogrid[: image.shape[0], : image.shape[1]]
        center_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2

        features["center_mean"] = np.mean(image[center_mask])
        features["periphery_mean"] = np.mean(image[~center_mask])
        features["center_periphery_ratio"] = features["center_mean"] / (
            features["periphery_mean"] + 1e-8
        )

        return features

    def extract_all_features(self, image):
        all_features = {}

        all_features.update(self.extract_intensity_features(image))
        all_features.update(self.extract_texture_features(image))
        all_features.update(self.extract_morphological_features(image))
        all_features.update(self.extract_spatial_features(image))

        return all_features


class LesionClassifier:
    def __init__(self, csv_path, images_dir):
        self.csv_path = csv_path
        self.images_dir = Path(images_dir)
        self.feature_extractor = LesionFeatureExtractor()
        self.label_encoders = {}
        self.models = {}
        self.feature_names = []

    def load_data(self):
        print("Loading CSV data...")
        self.df = pd.read_csv(self.csv_path)

        self.df = self.df.dropna(subset=["lesion_id"])

        print(f"Loaded {len(self.df)} records")
        print(f"Unique lesions: {self.df['lesion_id'].nunique()}")

    def prepare_target_variables(self):
        self.target_columns = [
            "lesion_occlusionLength",
            "lesion_dominance",
            "lesion_totalOcclusion",
            "lesion_heavyCalcification",
            "lesion_thrombus",
            "lesion_severeTortuosity",
        ]

        for col in self.target_columns:
            if col in self.df.columns:
                valid_mask = self.df[col].notna()
                if col == "lesion_totalOcclusion":
                    self.df.loc[valid_mask, col] = self.df.loc[valid_mask, col].astype(
                        str
                    )
                else:
                    self.df.loc[valid_mask, col] = self.df.loc[valid_mask, col].astype(
                        str
                    )

        print("Target variable distribution:")
        for col in self.target_columns:
            if col in self.df.columns:
                print(f"\n{col}:")
                print(self.df[col].value_counts().head())

    def load_images_and_extract_features(self):
        print("Loading images and extracting features...")

        features_list = []
        valid_indices = []

        for idx, row in self.df.iterrows():
            lesion_id = row["lesion_id"]
            if pd.isna(lesion_id):
                continue

            image_path = None
            for ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
                potential_path = self.images_dir / f"{lesion_id}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break

            if image_path is None:
                continue

            try:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                image = cv2.resize(image, self.feature_extractor.image_size)

                features = self.feature_extractor.extract_all_features(image)
                features_list.append(features)
                valid_indices.append(idx)

                if len(features_list) % 100 == 0:
                    print(f"Processed {len(features_list)} images...")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        if not features_list:
            raise ValueError("No valid images found!")

        self.features_df = pd.DataFrame(features_list)
        self.feature_names = list(self.features_df.columns)

        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        self.features_df = self.features_df.reset_index(drop=True)

        print(f"Successfully extracted features from {len(self.features_df)} images")
        print(f"Feature dimensions: {self.features_df.shape}")

    def train_model(self, target_column, test_size=0.2, random_state=42):
        if target_column not in self.df.columns:
            print(f"Target column {target_column} not found in data")
            return None

        valid_mask = self.df[target_column].notna()
        X = self.features_df[valid_mask].copy()
        y = self.df[valid_mask][target_column].copy()

        if len(X) < 10:
            print(f"Insufficient data for {target_column}: {len(X)} samples")
            return None

        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        self.label_encoders[target_column] = le

        print(f"\nTraining model for {target_column}")
        print(f"Classes: {le.classes_}")
        print(f"Samples: {len(X)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None,
        )

        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            loss_function="MultiClass" if len(le.classes_) > 2 else "Logloss",
            random_seed=random_state,
            verbose=False,
        )

        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.3f}")

        self.models[target_column] = {
            "model": model,
            "accuracy": accuracy,
            "feature_importance": dict(
                zip(self.feature_names, model.feature_importances_)
            ),
            "classification_report": classification_report(
                y_test, y_pred, target_names=le.classes_, output_dict=True
            ),
        }

        return self.models[target_column]

    def train_all_models(self):
        print("Training models for all target variables...")

        results = {}
        for target_col in self.target_columns:
            if target_col in self.df.columns:
                result = self.train_model(target_col)
                if result is not None:
                    results[target_col] = result

        return results

    def plot_feature_importance(self, target_column, top_n=15):
        if target_column not in self.models:
            print(f"No model found for {target_column}")
            return

        importance = self.models[target_column]["feature_importance"]
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]

        features, importances = zip(*sorted_features)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Features for {target_column}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_results_summary(self):
        if not self.models:
            print("No models trained yet")
            return

        targets = list(self.models.keys())
        accuracies = [self.models[t]["accuracy"] for t in targets]

        plt.figure(figsize=(12, 6))
        plt.bar(targets, accuracies)
        plt.title("Model Accuracy by Target Variable")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1)

        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.01, f"{acc:.3f}", ha="center")

        plt.tight_layout()
        plt.show()

    def generate_report(self):
        print("\n" + "=" * 80)
        print("LESION CLASSIFIER TRAINING REPORT")
        print("=" * 80)

        print(f"\nDataset Summary:")
        print(f"- Total records: {len(self.df)}")
        print(f"- Images processed: {len(self.features_df)}")
        print(f"- Features extracted: {len(self.feature_names)}")

        print(f"\nModels Trained:")
        for target, model_info in self.models.items():
            print(f"\n{target}:")
            print(f"  - Accuracy: {model_info['accuracy']:.3f}")

            importance = model_info["feature_importance"]
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            print(f"  - Top features: {[f[0] for f in top_features]}")

            report = model_info["classification_report"]
            if "accuracy" in report:
                print(f"  - Classes: {list(report.keys())[:-3]}")


def main():
    CSV_PATH = "Dataset-pruned/lesion_data.csv"
    IMAGES_DIR = "lesions"

    if not os.path.exists(CSV_PATH):
        print(f"CSV file not found: {CSV_PATH}")
        print(
            "Please update the CSV_PATH variable with the correct path to your lesion_data.csv file"
        )
        return

    if not os.path.exists(IMAGES_DIR):
        print(f"Images directory not found: {IMAGES_DIR}")
        print(
            "Please update the IMAGES_DIR variable with the correct path to your lesion images"
        )
        print(
            "Images should be named with their lesion_id (e.g., 'ca8b2bbe-6b6d-4c84-9e53-56f098d582c1.png')"
        )
        return

    try:
        classifier = LesionClassifier(CSV_PATH, IMAGES_DIR)

        classifier.load_data()
        classifier.prepare_target_variables()
        classifier.load_images_and_extract_features()

        results = classifier.train_all_models()

        classifier.plot_results_summary()

        for target in classifier.models.keys():
            classifier.plot_feature_importance(target)

        classifier.generate_report()

        if classifier.models:
            importance_data = []
            for target, model_info in classifier.models.items():
                for feature, importance in model_info["feature_importance"].items():
                    importance_data.append(
                        {"target": target, "feature": feature, "importance": importance}
                    )

            importance_df = pd.DataFrame(importance_data)
            importance_df.to_csv("feature_importance.csv", index=False)
            print(f"\nFeature importance saved to 'feature_importance.csv'")

    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
