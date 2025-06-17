import pandas as pd
import numpy as np


def analyze_lesion_characteristics(csv_file):
    df = pd.read_csv(csv_file)

    lesion_characteristics = [
        "lesion_dominance",
        "lesion_section",
        "lesion_totalOcclusion",
        "lesion_firstSegmentOfOcclusion",
        "lesion_firstSegmentAfterOcclusion",
        "lesion_occlusionAge",
        "lesion_sideBranch",
        "lesion_bluntStump",
        "lesion_bridging",
        "lesion_trifurcation",
        "lesion_trifurcationType",
        "lesion_bifurcation",
        "lesion_bifurcationType",
        "lesion_bifurcationAngle",
        "lesion_aortoOstialStenosis",
        "lesion_occlusionLength",
        "lesion_heavyCalcification",
        "lesion_thrombus",
        "lesion_severeTortuosity",
    ]

    print("LESION CHARACTERISTICS ANALYSIS")
    print("=" * 50)

    lesion_data = df[df["lesion_id"].notna()]
    total_lesions = len(lesion_data)

    print(f"Total number of lesions: {total_lesions}")
    print(f"Total number of rows: {len(df)}")
    print(f"Rows without lesion data: {len(df) - total_lesions}")
    print("\n")

    for characteristic in lesion_characteristics:
        print(f"--- {characteristic} ---")

        if characteristic in lesion_data.columns:
            non_null_data = lesion_data[characteristic].dropna()
            null_count = lesion_data[characteristic].isna().sum()

            print(f"Non-null values: {len(non_null_data)}")
            print(f"Null/missing values: {null_count}")
            print(f"Fill rate: {len(non_null_data)/total_lesions*100:.1f}%")

            if len(non_null_data) > 0:
                value_counts = non_null_data.value_counts().sort_index()

                print(f"Unique values: {len(value_counts)}")
                print("Value distribution:")

                for value, count in value_counts.items():
                    percentage = count / len(non_null_data) * 100
                    print(f"  '{value}': {count} ({percentage:.1f}%)")
            else:
                print("No non-null values found")
        else:
            print(f"Column '{characteristic}' not found in dataset")

        print()


def main():
    try:
        analyze_lesion_characteristics("Dataset-pruned/lesion_data.csv")

    except FileNotFoundError:
        print("Error: lesion_data.csv file not found in the current directory")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
