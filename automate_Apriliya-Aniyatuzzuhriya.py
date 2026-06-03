"""
automate_ZaldiAbdulHadi.py
==========================
File otomatisasi preprocessing dataset Titanic.
Mengembalikan data yang sudah siap dilatih (train & test split).

Penggunaan:
    python automate_ZaldiAbdulHadi.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ── Konfigurasi ────────────────────────────────────────────────
INPUT_PATH  = "titanic_raw.csv"
OUTPUT_DIR  = "titanic_preprocessing"
TEST_SIZE   = 0.2
RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════
# FUNGSI-FUNGSI PREPROCESSING
# ══════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """
    Memuat dataset Titanic dari file CSV.

    Parameters
    ----------
    path : str
        Path ke file CSV raw.

    Returns
    -------
    pd.DataFrame
        DataFrame hasil load.
    """
    print(f"[1/7] Memuat dataset dari '{path}' ...")
    df = pd.read_csv(path)
    print(f"      Ukuran dataset: {df.shape[0]} baris x {df.shape[1]} kolom")
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus kolom yang tidak relevan untuk pemodelan.

    Kolom yang dihapus:
    - deck        : missing value >75%
    - embark_town : redundan dengan 'embarked'
    - who         : redundan dengan 'sex' dan 'age'
    - alive       : redundan dengan 'survived'
    - adult_male  : redundan dengan 'sex'
    - alone       : bisa diturunkan dari sibsp + parch

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    print("[2/7] Menghapus kolom tidak relevan ...")
    kolom_hapus = ['deck', 'embark_town', 'who', 'alive', 'adult_male', 'alone']
    # Hanya hapus kolom yang benar-benar ada di dataframe
    kolom_hapus = [c for c in kolom_hapus if c in df.columns]
    df = df.drop(columns=kolom_hapus)
    print(f"      Kolom dihapus : {kolom_hapus}")
    print(f"      Kolom tersisa : {list(df.columns)}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values pada dataset.

    Strategi:
    - 'age'      : diisi median per kelas tiket (pclass)
    - 'embarked' : diisi modus (nilai paling sering)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    print("[3/7] Menangani missing values ...")
    before = df.isnull().sum().sum()

    # Age → median per pclass
    if 'age' in df.columns:
        df['age'] = df.groupby('pclass')['age'].transform(
            lambda x: x.fillna(x.median())
        )

    # Embarked → modus
    if 'embarked' in df.columns:
        df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

    after = df.isnull().sum().sum()
    print(f"      Missing values: {before} → {after}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari dataset.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    print("[4/7] Menghapus duplikat ...")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"      Duplikat dihapus: {before - after} baris")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mendeteksi dan menangani outlier menggunakan metode IQR (Capping/Winsorization).

    Fitur yang dicek: age, fare, sibsp, parch

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    print("[5/7] Menangani outlier (IQR Capping) ...")
    fitur_outlier = ['age', 'fare', 'sibsp', 'parch']
    fitur_outlier = [c for c in fitur_outlier if c in df.columns]

    for col in fitur_outlier:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outlier = len(df[(df[col] < lower) | (df[col] > upper)])
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"      {col:8s}: {n_outlier} outlier di-capping ke [{lower:.2f}, {upper:.2f}]")

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan encoding pada fitur kategorikal dan feature engineering.

    - 'sex'      : LabelEncoder (female=0, male=1)
    - 'embarked' : LabelEncoder (C=0, Q=1, S=2)
    - 'family_size' : fitur baru = sibsp + parch + 1

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    print("[6/7] Encoding fitur kategorikal & feature engineering ...")
    le = LabelEncoder()

    if 'sex' in df.columns:
        df['sex_encoded'] = le.fit_transform(df['sex'])
        print(f"      sex encoding    : {dict(zip(le.classes_, le.transform(le.classes_)))}")

    if 'embarked' in df.columns:
        df['embarked_encoded'] = le.fit_transform(df['embarked'])
        print(f"      embarked encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    if 'sibsp' in df.columns and 'parch' in df.columns:
        df['family_size'] = df['sibsp'] + df['parch'] + 1
        print("      family_size ditambahkan (sibsp + parch + 1)")

    return df


def scale_and_split(df: pd.DataFrame,
                    test_size: float = TEST_SIZE,
                    random_state: int = RANDOM_STATE):
    """
    Melakukan standarisasi fitur numerik dan train-test split.

    Fitur model:
        pclass, age, sibsp, parch, fare,
        sex_encoded, embarked_encoded, family_size

    Fitur yang di-scale: age, fare, family_size

    Parameters
    ----------
    df           : pd.DataFrame
    test_size    : float, proporsi data test (default 0.2)
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    """
    print("[7/7] Standarisasi & train-test split ...")

    fitur_model = ['pclass', 'age', 'sibsp', 'parch', 'fare',
                   'sex_encoded', 'embarked_encoded', 'family_size']
    fitur_model = [c for c in fitur_model if c in df.columns]

    X = df[fitur_model].copy()
    y = df['survived'].copy()

    # Standarisasi fitur numerik kontinu
    fitur_scale = [c for c in ['age', 'fare', 'family_size'] if c in X.columns]
    scaler = StandardScaler()
    X[fitur_scale] = scaler.fit_transform(X[fitur_scale])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"      Fitur digunakan : {fitur_model}")
    print(f"      Fitur di-scale  : {fitur_scale}")
    print(f"      Total data      : {len(X)}")
    print(f"      Data training   : {len(X_train)} ({int((1-test_size)*100)}%)")
    print(f"      Data testing    : {len(X_test)} ({int(test_size*100)}%)")

    return X_train, X_test, y_train, y_test


def save_outputs(X_train, X_test, y_train, y_test,
                 output_dir: str = OUTPUT_DIR) -> None:
    """
    Menyimpan hasil preprocessing ke folder output dalam format CSV.

    File yang disimpan:
    - titanic_preprocessed.csv : full dataset (X + y)
    - titanic_train.csv        : data training
    - titanic_test.csv         : data testing

    Parameters
    ----------
    X_train, X_test, y_train, y_test : hasil dari scale_and_split()
    output_dir : str, folder tujuan penyimpanan
    """
    os.makedirs(output_dir, exist_ok=True)

    # Full dataset
    full = pd.concat([
        pd.concat([X_train, X_test]).reset_index(drop=True),
        pd.concat([y_train, y_test]).reset_index(drop=True)
    ], axis=1)

    train_df = pd.concat([
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True)
    ], axis=1)

    test_df = pd.concat([
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True)
    ], axis=1)

    full.to_csv(f"{output_dir}/titanic_preprocessed.csv", index=False)
    train_df.to_csv(f"{output_dir}/titanic_train.csv", index=False)
    test_df.to_csv(f"{output_dir}/titanic_test.csv", index=False)

    print(f"\n✅ Hasil preprocessing disimpan ke folder '{output_dir}/'")
    print(f"   - titanic_preprocessed.csv ({len(full)} baris)")
    print(f"   - titanic_train.csv        ({len(train_df)} baris)")
    print(f"   - titanic_test.csv         ({len(test_df)} baris)")


# ══════════════════════════════════════════════════════════════
# FUNGSI UTAMA
# ══════════════════════════════════════════════════════════════

def preprocess(input_path: str = INPUT_PATH,
               output_dir: str = OUTPUT_DIR,
               test_size: float = TEST_SIZE,
               random_state: int = RANDOM_STATE):
    """
    Fungsi utama yang menjalankan seluruh pipeline preprocessing secara otomatis.

    Parameters
    ----------
    input_path   : str, path ke file CSV raw
    output_dir   : str, folder untuk menyimpan hasil
    test_size    : float, proporsi data test
    random_state : int, seed untuk reprodusibilitas

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    print("=" * 55)
    print(" PIPELINE PREPROCESSING TITANIC DATASET")
    print("=" * 55)

    df = load_data(input_path)
    df = drop_irrelevant_columns(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = encode_features(df)

    X_train, X_test, y_train, y_test = scale_and_split(
        df, test_size=test_size, random_state=random_state
    )

    save_outputs(X_train, X_test, y_train, y_test, output_dir)

    print("\n✅ Preprocessing selesai! Data siap dilatih.")
    print("=" * 55)

    return X_train, X_test, y_train, y_test


# ── Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess()

    print(f"\nShape X_train : {X_train.shape}")
    print(f"Shape X_test  : {X_test.shape}")
    print(f"\nSample X_train:")
    print(X_train.head())