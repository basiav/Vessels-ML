import pandas as pd
import os
import cv2
from ultralytics import YOLO
import yaml
from sklearn.model_selection import train_test_split
import shutil


class LesionYOLOTrainer:
    def __init__(
        self,
        base_images_dir="Dataset/base_images",
        lesion_data_path="Dataset/lesion_data.csv",
    ):
        self.base_images_dir = base_images_dir
        self.lesion_data_path = lesion_data_path
        self.dataset_dir = "yolo_dataset"
        self.lesion_df = None

    def load_data(self):
        print("Loading lesion data...")
        self.lesion_df = pd.read_csv(self.lesion_data_path)
        print(f"Loaded {len(self.lesion_df)} lesion annotations")

    def create_dataset_structure(self):
        print("Creating dataset structure...")

        # Create directories
        for split in ["train", "val"]:
            os.makedirs(f"{self.dataset_dir}/{split}/images", exist_ok=True)
            os.makedirs(f"{self.dataset_dir}/{split}/labels", exist_ok=True)

    def convert_to_yolo_format(self, x, y, width, height, img_width, img_height):
        center_x = x + width / 2
        center_y = y + height / 2

        norm_center_x = center_x / img_width
        norm_center_y = center_y / img_height
        norm_width = width / img_width
        norm_height = height / img_height

        return norm_center_x, norm_center_y, norm_width, norm_height

    def process_annotations(self):
        print("Processing annotations...")

        # Group lesions by image
        grouped = self.lesion_df.groupby(["image_id", "frame"])

        image_annotations = {}

        for (image_id, frame), group in grouped:
            image_filename = f"{image_id}_{frame}.png"
            image_path = os.path.join(self.base_images_dir, image_filename)

            if not os.path.exists(image_path):
                print(f"Warning: Image {image_filename} not found, skipping...")
                continue

            # Load image to get dimensions
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not load image {image_filename}, skipping...")
                continue

            img_height, img_width = img.shape[:2]

            annotations = []
            for _, row in group.iterrows():
                x = row["lesion_x"]
                y = row["lesion_y"]
                width = row["lesion_width"]
                height = row["lesion_height"]

                norm_cx, norm_cy, norm_w, norm_h = self.convert_to_yolo_format(
                    x, y, width, height, img_width, img_height
                )

                annotations.append(
                    f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}"
                )

            image_annotations[image_filename] = annotations

        return image_annotations

    def split_dataset(self, image_annotations, train_ratio=0.8):
        print("Splitting dataset...")

        image_files = list(image_annotations.keys())
        train_files, val_files = train_test_split(
            image_files, train_size=train_ratio, random_state=42
        )

        print(f"Training images: {len(train_files)}")
        print(f"Validation images: {len(val_files)}")

        return train_files, val_files

    def copy_files_and_create_labels(self, image_annotations, train_files, val_files):
        print("Copying files and creating labels...")

        for split, files in [("train", train_files), ("val", val_files)]:
            for image_file in files:
                # Copy image
                src_image = os.path.join(self.base_images_dir, image_file)
                dst_image = os.path.join(self.dataset_dir, split, "images", image_file)
                shutil.copy2(src_image, dst_image)

                # Create label file
                label_file = image_file.replace(".png", ".txt")
                label_path = os.path.join(self.dataset_dir, split, "labels", label_file)

                with open(label_path, "w") as f:
                    for annotation in image_annotations[image_file]:
                        f.write(annotation + "\n")

    def create_yaml_config(self):
        config = {
            "path": os.path.abspath(self.dataset_dir),
            "train": "train/images",
            "val": "val/images",
            "nc": 1,
            "names": ["lesion"],
        }

        config_path = os.path.join(self.dataset_dir, "dataset.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Created YAML config at {config_path}")
        return config_path

    def prepare_dataset(self):
        self.load_data()
        self.create_dataset_structure()

        image_annotations = self.process_annotations()
        train_files, val_files = self.split_dataset(image_annotations)
        self.copy_files_and_create_labels(image_annotations, train_files, val_files)

        config_path = self.create_yaml_config()
        return config_path

    def train_model(self, config_path, model_size="yolov8n", epochs=100, imgsz=512):
        print(f"Starting YOLO training with {model_size}...")

        # Load a pre-trained model
        model = YOLO(f"{model_size}.pt")

        # Train the model
        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=imgsz,
            patience=10,
            save=True,
            # device="CUDA", :-(
            device="cpu",
            workers=4,
            batch=16,
            name="lesion_detection",
        )

        print("Training completed!")
        return model, results

    def evaluate_model(self, model):
        print("Evaluating model...")

        # Validate the model
        metrics = model.val()

        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")

        return metrics

    def predict_sample(self, model, image_path, conf_threshold=0.25):
        print(f"Making prediction on {image_path}...")

        results = model.predict(
            image_path, conf=conf_threshold, save=True, show_labels=True, show_conf=True
        )

        return results


def main():
    # Initialize trainer
    trainer = LesionYOLOTrainer()

    # Prepare dataset
    config_path = trainer.prepare_dataset()

    # Train model
    # yolov8n -> nano version

    model, results = trainer.train_model(
        config_path=config_path,
        model_size="yolov8n",
        epochs=100,
        imgsz=512,
    )

    # Evaluate model
    metrics = trainer.evaluate_model(model)

    print("Evaluation metrics:")
    print(metrics)

    # Test on a sample image
    sample_images = os.listdir(trainer.base_images_dir)
    if sample_images:
        sample_path = os.path.join(trainer.base_images_dir, sample_images[0])
        predictions = trainer.predict_sample(model, sample_path)

    print(f"Model saved")


if __name__ == "__main__":
    main()
