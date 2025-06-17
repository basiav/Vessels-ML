import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from pathlib import Path
import argparse


class LesionVisualizationTool:
    def __init__(self, csv_path, images_dir):
        self.csv_path = csv_path
        self.images_dir = Path(images_dir)
        self.lesion_data = None
        self.load_data()

    def load_data(self):
        print(f"Loading lesion data from {self.csv_path}")
        self.lesion_data = pd.read_csv(self.csv_path)
        self.lesion_data = self.lesion_data.dropna(
            subset=["lesion_x", "lesion_y", "lesion_width", "lesion_height"]
        )

        print(f"Loaded {len(self.lesion_data)} lesion annotations")
        print(f"Unique images: {self.lesion_data['image_id'].nunique()}")
        print(f"Unique frames: {self.lesion_data['frame'].nunique()}")

    def get_image_path(self, image_id, frame):
        image_name = f"{image_id}_{frame}.png"
        return self.images_dir / image_name

    def visualize_single_image(self, image_id, frame, save_path=None, show_plot=True):
        img_path = self.get_image_path(image_id, frame)

        if not img_path.exists():
            print(f"Image not found: {img_path}")
            return False

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load image: {img_path}")
            return False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lesions = self.lesion_data[
            (self.lesion_data["image_id"] == image_id)
            & (self.lesion_data["frame"] == frame)
        ]

        if len(lesions) == 0:
            print(f"No lesions found for image {image_id}, frame {frame}")
            return False

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(image)

        colors = [
            "red",
            "green",
            "blue",
            "yellow",
            "cyan",
            "magenta",
            "orange",
            "purple",
        ]

        for idx, (_, lesion) in enumerate(lesions.iterrows()):
            x = lesion["lesion_x"]
            y = lesion["lesion_y"]
            width = lesion["lesion_width"]
            height = lesion["lesion_height"]

            color = colors[idx % len(colors)]
            rect = patches.Rectangle(
                (x, y), width, height, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            lesion_info = f"ID: {lesion['lesion_id'][:8]}...\n"
            lesion_info += f"Size: {width}x{height}\n"
            lesion_info += f"Pos: ({x:.0f},{y:.0f})\n"
            lesion_info += f"Section: {lesion['lesion_section']}"

            ax.text(
                x,
                y - 10,
                lesion_info,
                fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        ax.set_title(
            f"Image: {image_id}\nFrame: {frame} | Lesions: {len(lesions)}",
            fontsize=12,
            fontweight="bold",
        )
        ax.axis("off")

        img_height, img_width = image.shape[:2]
        fig.suptitle(
            f"Lesion Visualization - Image Size: {img_width}x{img_height}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return True

    def generate_all_labeled_images(self, save_dir="labeled"):
        unique_combos = self.lesion_data[["image_id", "frame"]].drop_duplicates()

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        print(f"Generating {len(unique_combos)} labeled images...")

        success_count = 0
        failed_count = 0

        for idx, (_, row) in enumerate(unique_combos.iterrows()):
            image_id = row["image_id"]
            frame = row["frame"]

            save_path = save_dir / f"{image_id}_{frame}_labeled.png"

            print(f"Processing {idx+1}/{len(unique_combos)}: {image_id}_{frame}")
            success = self.visualize_single_image(
                image_id, frame, save_path=save_path, show_plot=False
            )

            if success:
                success_count += 1
            else:
                failed_count += 1

        print(f"\nCompleted: {success_count} success, {failed_count} failed")
        print(f"Labeled images saved to: {save_dir.absolute()}")

    def visualize_random_samples(self, num_samples=5, save_dir=None):
        unique_combos = self.lesion_data[["image_id", "frame"]].drop_duplicates()

        if len(unique_combos) > num_samples:
            sampled = unique_combos.sample(n=num_samples, random_state=42)
        else:
            sampled = unique_combos

        print(f"Visualizing {len(sampled)} random samples...")

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        for idx, (_, row) in enumerate(sampled.iterrows()):
            image_id = row["image_id"]
            frame = row["frame"]

            save_path = None
            if save_dir:
                save_path = save_dir / f"{image_id}_{frame}_labeled.png"

            print(f"\nProcessing sample {idx+1}: {image_id}, frame {frame}")
            success = self.visualize_single_image(
                image_id, frame, save_path=save_path, show_plot=False
            )

            if not success:
                print(f"Failed to process sample {idx+1}")

    def visualize_specific_images(self, image_frame_pairs):
        for image_id, frame in image_frame_pairs:
            print(f"\nVisualizing {image_id}, frame {frame}")
            self.visualize_single_image(image_id, frame, show_plot=True)

    def analyze_annotations(self):
        print("\n=== ANNOTATION ANALYSIS ===")

        # Basic statistics
        print(f"Total annotations: {len(self.lesion_data)}")
        print(f"Unique images: {self.lesion_data['image_id'].nunique()}")
        print(f"Unique frames: {self.lesion_data['frame'].nunique()}")

        # Bounding box statistics
        print(f"\nBounding Box Statistics:")
        print(
            f"Width - Min: {self.lesion_data['lesion_width'].min():.1f}, "
            f"Max: {self.lesion_data['lesion_width'].max():.1f}, "
            f"Mean: {self.lesion_data['lesion_width'].mean():.1f}"
        )
        print(
            f"Height - Min: {self.lesion_data['lesion_height'].min():.1f}, "
            f"Max: {self.lesion_data['lesion_height'].max():.1f}, "
            f"Mean: {self.lesion_data['lesion_height'].mean():.1f}"
        )

        # Position statistics
        print(f"\nPosition Statistics:")
        print(
            f"X - Min: {self.lesion_data['lesion_x'].min():.1f}, "
            f"Max: {self.lesion_data['lesion_x'].max():.1f}, "
            f"Mean: {self.lesion_data['lesion_x'].mean():.1f}"
        )
        print(
            f"Y - Min: {self.lesion_data['lesion_y'].min():.1f}, "
            f"Max: {self.lesion_data['lesion_y'].max():.1f}, "
            f"Mean: {self.lesion_data['lesion_y'].mean():.1f}"
        )

        print(f"\n=== POTENTIAL ISSUES ===")
        small_boxes = self.lesion_data[
            (self.lesion_data["lesion_width"] < 10)
            | (self.lesion_data["lesion_height"] < 10)
        ]
        print(f"Very small bounding boxes (width or height < 10): {len(small_boxes)}")

        large_boxes = self.lesion_data[
            (self.lesion_data["lesion_width"] > 200)
            | (self.lesion_data["lesion_height"] > 200)
        ]
        print(f"Very large bounding boxes (width or height > 200): {len(large_boxes)}")

        missing_files = 0
        for _, row in (
            self.lesion_data[["image_id", "frame"]].drop_duplicates().iterrows()
        ):
            img_path = self.get_image_path(row["image_id"], row["frame"])
            if not img_path.exists():
                missing_files += 1
        print(f"Missing image files: {missing_files}")

        return {
            "small_boxes": small_boxes,
            "large_boxes": large_boxes,
            "missing_files": missing_files,
        }

    def create_annotation_summary(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Width distribution
        axes[0, 0].hist(
            self.lesion_data["lesion_width"], bins=50, alpha=0.7, color="blue"
        )
        axes[0, 0].set_title("Lesion Width Distribution")
        axes[0, 0].set_xlabel("Width (pixels)")
        axes[0, 0].set_ylabel("Frequency")

        # Height distribution
        axes[0, 1].hist(
            self.lesion_data["lesion_height"], bins=50, alpha=0.7, color="green"
        )
        axes[0, 1].set_title("Lesion Height Distribution")
        axes[0, 1].set_xlabel("Height (pixels)")
        axes[0, 1].set_ylabel("Frequency")

        # Position scatter plot
        axes[1, 0].scatter(
            self.lesion_data["lesion_x"],
            self.lesion_data["lesion_y"],
            alpha=0.6,
            c="red",
            s=20,
        )
        axes[1, 0].set_title("Lesion Position Distribution")
        axes[1, 0].set_xlabel("X Position")
        axes[1, 0].set_ylabel("Y Position")
        axes[1, 0].invert_yaxis()

        # Area distribution
        areas = self.lesion_data["lesion_width"] * self.lesion_data["lesion_height"]
        axes[1, 1].hist(areas, bins=50, alpha=0.7, color="orange")
        axes[1, 1].set_title("Lesion Area Distribution")
        axes[1, 1].set_xlabel("Area (pixels²)")
        axes[1, 1].set_ylabel("Frequency")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved annotation summary to: {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize lesion annotations on medical images"
    )
    parser.add_argument(
        "--csv",
        default="./Dataset-pruned/lesion_data.csv",
        help="Path to lesion data CSV file",
    )
    parser.add_argument(
        "--images",
        default="./Dataset-pruned/base_images",
        help="Path to images directory",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "random", "specific"],
        default="all",
        help="Visualization mode",
    )
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of random samples to visualize"
    )
    parser.add_argument(
        "--save_dir", default="labeled", help="Directory to save visualizations"
    )
    parser.add_argument("--image_id", help="Specific image ID to visualize")
    parser.add_argument("--frame", type=int, help="Specific frame to visualize")

    args = parser.parse_args()
    viz = LesionVisualizationTool(args.csv, args.images)

    if args.mode == "all":
        viz.generate_all_labeled_images(args.save_dir)

    elif args.mode == "specific":
        if args.image_id and args.frame is not None:
            viz.visualize_single_image(args.image_id, args.frame)
        else:
            print("Please provide --image_id and --frame for specific mode")

    else:
        viz.visualize_random_samples(args.samples, args.save_dir)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Running in default mode - generating all labeled images...")

        viz = LesionVisualizationTool(
            "./Dataset-pruned/lesion_data.csv",
            "./Dataset-pruned/base_images",
        )

        viz.generate_all_labeled_images("labeled-pruned")

    else:
        main()
