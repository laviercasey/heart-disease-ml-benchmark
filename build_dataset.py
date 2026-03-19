import pandas as pd
import numpy as np

DATA_DIR    = "./data/"
OUTPUT      = "./data/heart_combined_clean.csv"
OUTPUT_SRC  = "./data/heart_combined_with_source.csv"

COLUMNS = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_bs", "resting_ecg", "max_hr", "exercise_angina",
    "oldpeak", "st_slope", "target",
]
COLUMNS_SRC = COLUMNS + ["source"]

INT_COLS   = ["age", "sex", "chest_pain_type", "fasting_bs",
              "resting_ecg", "exercise_angina", "st_slope", "target"]
FLOAT_COLS = ["resting_bp", "cholesterol", "max_hr", "oldpeak"]
CAT_COLS   = ["sex", "chest_pain_type", "fasting_bs",
              "resting_ecg", "exercise_angina", "st_slope"]


def prepare_heart(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "cp": "chest_pain_type", "trestbps": "resting_bp",
        "chol": "cholesterol",   "fbs": "fasting_bs",
        "restecg": "resting_ecg","thalach": "max_hr",
        "exang": "exercise_angina", "slope": "st_slope",
    })
    df["chest_pain_type"] += 1
    return df


def prepare_cleveland(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "cp": "chest_pain_type", "trestbps": "resting_bp",
        "chol": "cholesterol",   "fbs": "fasting_bs",
        "restecg": "resting_ecg","thalach": "max_hr",
        "exang": "exercise_angina", "slope": "st_slope",
        "condition": "target",
    })
    df["chest_pain_type"] += 1
    return df


def prepare_uci(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "cp": "chest_pain_type", "trestbps": "resting_bp",
        "chol": "cholesterol",   "fbs": "fasting_bs",
        "restecg": "resting_ecg","thalch": "max_hr",
        "exang": "exercise_angina", "slope": "st_slope",
        "num": "target",
    })
    df["target"] = (df["target"] > 0).astype(int)
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    df["chest_pain_type"] = df["chest_pain_type"].map({
        "typical angina": 1, "atypical angina": 2,
        "non-anginal": 3,    "asymptomatic": 4,
    })
    df["resting_ecg"] = df["resting_ecg"].map({
        "normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2,
    })
    df["exercise_angina"] = df["exercise_angina"].map({True: 1, False: 0})
    df["st_slope"] = df["st_slope"].map({
        "upsloping": 1, "flat": 2, "downsloping": 3,
    })
    return df.drop(columns=["id", "dataset"], errors="ignore")


def prepare_statlog(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "chest pain type": "chest_pain_type",
        "resting bp s":    "resting_bp",
        "fasting blood sugar": "fasting_bs",
        "resting ecg":     "resting_ecg",
        "max heart rate":  "max_hr",
        "exercise angina": "exercise_angina",
        "ST slope":        "st_slope",
    })


def impute(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if col not in CAT_COLS and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CAT_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in INT_COLS:
        df[col] = df[col].astype(int)
    for col in FLOAT_COLS:
        df[col] = df[col].astype(float)
    return df


def build() -> None:
    heart    = pd.read_csv(DATA_DIR + "heart_kaggle.csv")
    cleveland = pd.read_csv(DATA_DIR + "heart_cleveland.csv")
    uci      = pd.read_csv(DATA_DIR + "heart_uci_multicenter.csv")
    statlog  = pd.read_csv(DATA_DIR + "heart_statlog.csv")

    heart_prep    = prepare_heart(heart)[COLUMNS]
    heart_prep["source"] = "kaggle"

    cleveland_prep = prepare_cleveland(cleveland)[COLUMNS]
    cleveland_prep["source"] = "cleveland"

    uci_prep = prepare_uci(uci)
    source_map = {
        "Cleveland": "cleveland", "Hungary": "hungary",
        "VA Long Beach": "va",    "Switzerland": "switzerland",
    }
    uci_prep["source"] = (
        uci["dataset"].map(source_map).fillna("uci")
        if "dataset" in uci.columns else "uci_other"
    )
    uci_prep = uci_prep[COLUMNS_SRC]

    statlog_prep = prepare_statlog(statlog)[COLUMNS]
    statlog_prep["source"] = "statlog"

    combined = pd.concat(
        [heart_prep, cleveland_prep, uci_prep, statlog_prep],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(subset=COLUMNS)
    combined = impute(combined)
    combined = cast_types(combined)

    combined[COLUMNS].to_csv(OUTPUT, index=False)
    combined[COLUMNS_SRC].to_csv(OUTPUT_SRC, index=False)

    print(f"Saved {len(combined)} rows → {OUTPUT}")


if __name__ == "__main__":
    build()
