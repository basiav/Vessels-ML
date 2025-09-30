import pandas as pd
import os
import cv2
import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from time import time
from PIL import Image
from sklearn.model_selection import train_test_split

ORIGINAL_DATASET_DIR = "data/Dataset"
PRUNED_DATASET_DIR = "data/Dataset-pruned"
ORIGINAL_IMGS_DIR = ORIGINAL_DATASET_DIR + "/base_images"
PRUNED_IMGS_DIR = PRUNED_DATASET_DIR + "/base_images"

ALLOWED_IMG_FORMATS = [".png", ".jpg", ".jpeg"]

DEFAULT_IMG_SIZE = (512, 512)
THRESHOLD = 128


def filename_to_bin_img_array(filename: str, size: tuple[int, int] | None = None, threshold: int = THRESHOLD) -> np.ndarray:
    # path = os.path.join(dataset_dir, filename)
    img = Image.open(filename).convert("L") # grayscale

    if size is not None:
        img = img.resize(size, Image.NEAREST) # resize

    img_arr = np.array(img)
    bin_arr = (img_arr >= threshold).astype(np.uint8)
    
    return bin_arr

def get_img_array(filenames: list[str], size: tuple[int, int] | None = None) -> list[np.ndarray]:
    img_array = []
    for filename in filenames:
        # try:
        #     bin_arr = filename_to_bin_img_array(filename, size)
        #     img_array.append(bin_arr)
        # except FileNotFoundError:
        #     print(f"Warning: file not found, skipping {filename}")
        #     continue
        # filename = os.path.join(dataset_dir, filename)
        bin_arr = filename_to_bin_img_array(filename, size)
        img_array.append(bin_arr)

    return img_array

def filter_uneven_shapes(img_array: list[np.ndarray], img_size_standard: tuple[int, int] = DEFAULT_IMG_SIZE, threshold: int = THRESHOLD) -> np.ndarray: # list[np.ndarray]:
    imgs_filtered = []
    for img in img_array:
        if img.shape != img_size_standard:
            print(f"Resizing img: {img.shape}")
            img = Image.fromarray(img*255)
            img = img.convert("L")          # grayscale
            img = img.resize(img_size_standard, Image.NEAREST) # resize
            img_arr = np.array(img)
            bin_arr = (img_arr >= threshold).astype(np.uint8)
            imgs_filtered.append(bin_arr)
        else:
            imgs_filtered.append(img)
        
    imgs_filtered = np.array(imgs_filtered)
    
    return imgs_filtered


def load_lesion_df(dataset_dir: str = ORIGINAL_DATASET_DIR) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(dataset_dir, "lesion_data.csv"))
    return df


def get_all_imgs_filenames_from_dir_list(images_dir: str = ORIGINAL_IMGS_DIR) -> list[str]:
    filenames_in_dir = [f for f in os.listdir(images_dir) if any(f.endswith(ext) for ext in ALLOWED_IMG_FORMATS)]
    return filenames_in_dir


def filter_non_lesion_entries(df: pd.DataFrame) -> pd.DataFrame:
    lesions_present_df = df[~df["lesion_id"].isna()]
    assert lesions_present_df.shape == df[~df["lesion_width"].isna()].shape
    assert lesions_present_df.shape == df[~df["lesion_x"].isna()].shape
    return lesions_present_df

def get_lesion_filenames_list(df: pd.DataFrame) -> list[str]:
    lesion_df_filenames = (df["image_id"].astype(str) + "_" + df["frame"].astype(str) + ".png").values
    lesion_df_filenames_list = lesion_df_filenames.tolist()
    return lesion_df_filenames_list


# def get_lesion_present_filenames(df: pd.DataFrame) -> list[str]:
#     lesions_present_df = df[~df["lesion_id"].isna()]
#     assert lesions_present_df.shape == df[~df["lesion_width"].isna()].shape
#     assert lesions_present_df.shape == df[~df["lesion_x"].isna()].shape
#     return get_lesion_filenames_list(lesions_present_df)

# def get_lesion_absent_filenames(df: pd.DataFrame) -> list[str]:
#     lesions_present_df = df[~df["lesion_id"].isna()]
#     lesions_absent_df = df[df["lesion_id"].isna()]
#     assert pd.concat([lesions_present_df, lesions_absent_df]).shape == df.shape
#     return get_lesion_filenames_list(lesions_absent_df)


def split_lesions_present_absent(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lesions_present_df = df[~df["lesion_id"].isna()]
    lesions_absent_df = df[df["lesion_id"].isna()]
    assert pd.concat([lesions_present_df, lesions_absent_df]).shape == df.shape
    return lesions_present_df, lesions_absent_df

# def get_Y_train(filenames_in_dir: list[str], img_array: list[np.ndarray], lesion_df_filenames_list: list[str]) -> np.ndarray:
#     Y_train = []
#     for file_path, _ in zip(filenames_in_dir, img_array):
#         if file_path in lesion_df_filenames_list:
#             Y_train.append(1)
#         else:
#             Y_train.append(0)

#     Y_train = np.array(Y_train)
#     return Y_train


def find_duplicates(some_list: list) -> list:
    seen = set()
    duplicates = set()

    for x in some_list:
        if x in seen:
            duplicates.add(x)
        else:
            seen.add(x)

    return list(duplicates)

