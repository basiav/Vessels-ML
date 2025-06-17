import pandas as pd
import cv2
import numpy as np
import os
from pathlib import Path


def extract_lesion_frames(csv_file_path, images_folder_path, output_folder="lesions"):
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(csv_file_path)
    df_filtered = df.dropna(
        subset=["lesion_x", "lesion_y", "lesion_width", "lesion_height"]
    )

    print(f"Found {len(df_filtered)} valid lesion entries")

    for idx, row in df_filtered.iterrows():
        try:
            image_id = row["image_id"]
            frame_number = int(row["frame"])

            image_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm"]:
                potential_path = os.path.join(
                    images_folder_path, f"{image_id}_{frame_number}{ext}"
                )
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break

            if image_path is None:
                print(
                    f"Warning: Image file not found for image_id: {image_id}, frame: {frame_number}"
                )
                continue

            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not load image: {image_path}")
                    continue

                print(f"Processing image: {image_id}_frame_{frame_number}")

            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            lesion_x = int(row["lesion_x"])
            lesion_y = int(row["lesion_y"])
            lesion_width = int(row["lesion_width"])
            lesion_height = int(row["lesion_height"])
            lesion_id = row["lesion_id"]

            center_x = lesion_x + lesion_width // 2
            center_y = lesion_y + lesion_height // 2

            frame_size = 100
            half_frame = frame_size // 2

            start_x = max(0, center_x - half_frame)
            start_y = max(0, center_y - half_frame)
            end_x = min(image.shape[1], center_x + half_frame)
            end_y = min(image.shape[0], center_y + half_frame)

            lesion_frame = image[start_y:end_y, start_x:end_x]

            if lesion_frame.shape[0] < frame_size or lesion_frame.shape[1] < frame_size:
                padded_frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

                pad_y = (frame_size - lesion_frame.shape[0]) // 2
                pad_x = (frame_size - lesion_frame.shape[1]) // 2

                padded_frame[
                    pad_y : pad_y + lesion_frame.shape[0],
                    pad_x : pad_x + lesion_frame.shape[1],
                ] = lesion_frame

                lesion_frame = padded_frame

            output_filename = f"{lesion_id}.png"
            output_path = os.path.join(output_folder, output_filename)

            cv2.imwrite(output_path, lesion_frame)

            print(f"Saved lesion frame: {output_filename}")

        except Exception as e:
            print(f"Error processing lesion {row.get('lesion_id', 'unknown')}: {e}")
            continue


def main():
    csv_file_path = "Dataset-pruned/lesion_data.csv"
    images_folder_path = "Dataset-pruned/base_images"
    output_folder = "lesions"

    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        return

    if not os.path.exists(images_folder_path):
        print(f"Error: Images folder not found at {images_folder_path}")
        return

    extract_lesion_frames(csv_file_path, images_folder_path, output_folder)

    print("Lesion extraction completed!")


if __name__ == "__main__":
    main()
