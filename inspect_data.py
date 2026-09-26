import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

files = {
    "train_source1": "dataset/train/train_source1.tsv",
    "train_source2": "dataset/train/train_source2.tsv",
    "train_source3": "dataset/train/train_source3.tsv",
    "train_ground_truth": "dataset/train/train_ground_truth.tsv",
    "test_source1": "dataset/test/test_source1.tsv",
    "test_source2": "dataset/test/test_source2.tsv",
    "test_source3": "dataset/test/test_source3.tsv",
}

for name, path in files.items():
    print(f"\n{'='*70}\n{name}  ({path})\n{'='*70}")
    try:
        df = pd.read_csv(path, sep="\t")
        print("shape:", df.shape)
        print("columns:", list(df.columns))
        print("\ndtypes:\n", df.dtypes)
        print("\nmissing values:\n", df.isna().sum())
        print("\nsample rows:\n", df.head(5).to_string())
        if "country" in df.columns:
            print("\ncountry value counts:\n", df["country"].value_counts(dropna=False))
    except Exception as e:
        print("ERROR reading file:", e)