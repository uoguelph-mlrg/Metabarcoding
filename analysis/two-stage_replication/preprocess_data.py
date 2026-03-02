"""
Preprocess ecuador_training_data.csv and save train/val/test splits.
This creates the X_train, X_val, X_test, y_train, y_val, y_test CSV files.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse


# Features from utils.py
OBSERVATION_FEATURES = [
    "Excess",
    "trackingsl_specimen_count",
    "Bulk_Sample_wet_weight",
    "SumExcessSpecimens",
    "ExcessNumberTaxa",
    "total_reads_per_sample",
    "length_min_mm",
    "month",
    "avg_length_mm",
    "reads_per_specimen",
    "specimens_per_gram",
    "taxonomic_completeness",
    "read_consistency",
    "relative_read_abundance"
]

TAXONOMY_FEATURES = ["phylum", "class", "order", "family"]


def preprocess_and_save(
    input_path: str = "../../data/ecuador_training_data.csv",
    output_dir: str = "data",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    random_state: int = 42
):
    # === FEATURE ENGINEERING (MATCH ITERATION 6) ===
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    # 1. TEMPORAL FEATURES
    if 'collection_start_date' in df.columns:
        df['collection_date'] = pd.to_datetime(df['collection_start_date'])
        df['month'] = df['collection_date'].dt.month
    # 2. SIZE-BASED FEATURES
    if all(col in df.columns for col in ['length_min_mm', 'length_max_mm']):
        df['avg_length_mm'] = (df['length_min_mm'] + df['length_max_mm']) / 2
        df['length_range_mm'] = df['length_max_mm'] - df['length_min_mm']
        def size_category(avg_length):
            if pd.isna(avg_length):
                return 'Unknown'
            elif avg_length < 5:
                return 'Very_Small'
            elif avg_length < 15:
                return 'Small'
            elif avg_length < 30:
                return 'Medium'
            elif avg_length < 60:
                return 'Large'
            else:
                return 'Very_Large'
        df['size_category'] = df['avg_length_mm'].apply(size_category)
    # 3. SAMPLE EFFICIENCY FEATURES
    if all(col in df.columns for col in ['total_reads', 'trackingsl_specimen_count', 'Bulk_Sample_wet_weight']):
        df['reads_per_specimen'] = np.where(
            df['trackingsl_specimen_count'] > 0,
            df['total_reads'] / df['trackingsl_specimen_count'],
            np.nan
        )
        df['specimens_per_gram'] = np.where(
            df['Bulk_Sample_wet_weight'] > 0,
            df['trackingsl_specimen_count'] / df['Bulk_Sample_wet_weight'],
            np.nan
        )
        df['reads_per_gram'] = np.where(
            df['Bulk_Sample_wet_weight'] > 0,
            df['total_reads'] / df['Bulk_Sample_wet_weight'],
            np.nan
        )
    # 4. TAXONOMIC COMPLETENESS
    taxonomic_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    available_taxonomic = [level for level in taxonomic_levels if level in df.columns]
    if len(available_taxonomic) > 0:
        df['taxonomic_completeness'] = df[available_taxonomic].notna().sum(axis=1) / len(available_taxonomic)
    # 5. READ DISTRIBUTION FEATURES
    if all(col in df.columns for col in ['total_reads', 'avg_reads', 'min_reads', 'max_reads']):
        df['read_consistency'] = np.where(
            df['max_reads'] > 0,
            df['min_reads'] / df['max_reads'],
            np.nan
        )
        df['relative_read_abundance'] = np.where(
            df['total_reads_per_sample'] > 0,
            df['total_reads'] / df['total_reads_per_sample'],
            np.nan
        )
    
    # Rename columns to match expected format
    df = df.rename(columns={"sample-eventid": "sample_id"})
    
    # Parse date and extract day of year as numeric feature
    if "collection_start_date" in df.columns:
        df["collection_day"] = pd.to_datetime(
            df["collection_start_date"], format="%m/%d/%Y", errors="coerce"
        ).dt.dayofyear
        df["collection_day"] = df["collection_day"].fillna(0)
    else:
        df["collection_day"] = 0
    

    # Build index mappings
    unique_samples = df["sample_id"].unique()
    n_samples = len(unique_samples)
    unique_bins = df["bin_uri"].unique()
    print(f"Found {n_samples} samples, {len(unique_bins)} bins, {len(df)} observations")

    # Compute relative abundance (target)
    sample_totals = df.groupby("sample_id")["occurrences"].transform("sum")
    df["rel_abundance"] = df["occurrences"] / (sample_totals + 1e-10)

    # Normalize reads per sample (log-transformed)
    df["total_reads_norm"] = np.log1p(df["total_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2)
    df["avg_reads_norm"] = np.log1p(df["avg_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2)
    df["max_reads_norm"] = np.log1p(df["max_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2)
    df["min_reads_norm"] = np.log1p(df["min_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2)

    # Track dropped columns
    dropped_columns = [col for col in OBSERVATION_FEATURES + TAXONOMY_FEATURES if col not in df.columns]
    dropped_columns_log = os.path.join(output_dir, "dropped_columns_log.csv")
    pd.DataFrame({"dropped_column": dropped_columns}).to_csv(dropped_columns_log, index=False)

    # Get available feature columns
    feature_cols_present = [c for c in OBSERVATION_FEATURES if c in df.columns]
    taxonomy_cols_present = [c for c in TAXONOMY_FEATURES if c in df.columns]
    print(f"Observation features: {len(feature_cols_present)}")
    print(f"Taxonomy features: {len(taxonomy_cols_present)}")


    # --- ENCODING STRATEGY (MATCH ITERATION 6) ---
    from category_encoders import TargetEncoder
    # Identify categorical features (excluding taxonomy)
    categorical_features = [col for col in df.columns if df[col].dtype == 'object' and col not in taxonomy_cols_present and col not in ['sample_id', 'bin_uri']]
    # Taxonomic hierarchy for hierarchical encoding
    TAXONOMIC_HIERARCHY = ['phylum', 'class', 'order', 'family']
    available_hierarchy = [feat for feat in TAXONOMIC_HIERARCHY if feat in df.columns]
    encoding_strategy = []
    # Hierarchical encoding for taxonomy
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for i, level in enumerate(available_hierarchy):
        df[level] = df[level].fillna('Unknown')
        le = LabelEncoder()
        df[f'{level}_hierarchical'] = le.fit_transform(df[level])
        label_encoders[level] = le
        encoding_strategy.append({'feature': level, 'strategy': 'hierarchical', 'classes': list(le.classes_)})
    # Path encoding for full taxonomy
    if available_hierarchy:
        taxonomy_paths = df[available_hierarchy].astype(str).agg(' > '.join, axis=1)
        path_le = LabelEncoder()
        df['taxonomic_path'] = path_le.fit_transform(taxonomy_paths)
        label_encoders['taxonomic_path'] = path_le
        encoding_strategy.append({'feature': 'taxonomic_path', 'strategy': 'hierarchical', 'classes': list(path_le.classes_)})
    # One-hot, binary, target encoding for other categoricals
    from sklearn.preprocessing import OneHotEncoder
    for col in categorical_features:
        unique_count = df[col].nunique()
        if unique_count == 2:
            # Binary encoding
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
            label_encoders[col] = le
            encoding_strategy.append({'feature': col, 'strategy': 'binary', 'classes': list(le.classes_)})
        elif unique_count <= 10:
            # One-hot encoding
            dummies = pd.get_dummies(df[col].fillna('Unknown'), prefix=col, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            encoding_strategy.append({'feature': col, 'strategy': 'onehot', 'classes': list(df[col].unique())})
        else:
            # Target encoding
            te = TargetEncoder(smoothing=10, min_samples_leaf=20)
            df[f'{col}_target'] = te.fit_transform(df[col], df['rel_abundance'])
            encoding_strategy.append({'feature': col, 'strategy': 'target', 'classes': list(df[col].unique())})
    # Compose final feature list
    taxonomy_encoded_cols = [f'{level}_hierarchical' for level in available_hierarchy] + (['taxonomic_path'] if available_hierarchy else [])
    # Only keep size_category one-hot columns
    size_category_cols = [c for c in df.columns if c.startswith('size_category_')]
    # Only keep depth features if present
    depth_features = [f'{level}_depth' for level in TAXONOMY_FEATURES if f'{level}_depth' in df.columns]
    all_feature_cols = feature_cols_present + taxonomy_encoded_cols + depth_features + ['taxonomic_path'] + size_category_cols
    # Filter encoding_strategy to only those in Iteration 6
    encoding_strategy_filtered = [e for e in encoding_strategy if e['feature'] in TAXONOMY_FEATURES or e['feature'] == 'taxonomic_path' or e['feature'].startswith('size_category')]
    pd.DataFrame(encoding_strategy_filtered).to_csv(os.path.join(output_dir, "encoding_strategy.csv"), index=False)

    print(f"Total features: {len(all_feature_cols)}")

    # Build df_long with required columns + features
    base_cols = ["sample_id", "bin_uri", "occurrences", "rel_abundance"]
    df_long = df[base_cols + all_feature_cols].copy()


    # SYSTEMATIC MODULO-20 SPLIT BY SAMPLE_ID (MATCH ITERATION 6)
    # 1. Calculate total occurrences per sample_id
    eventid_occurrences = df.groupby('sample_id').agg({
        'occurrences': 'sum',
        'bin_uri': 'first'  # just to keep a reference
    }).reset_index()
    # 2. Sort by total occurrences (descending)
    eventid_occurrences = eventid_occurrences.sort_values('occurrences', ascending=False).reset_index(drop=True)
    # 3. Assign splits using modulo 20 pattern
    def assign_split(index):
        mod_val = index % 20
        if mod_val <= 13:
            return 'train'
        elif mod_val <= 16:
            return 'val'
        else:
            return 'test'
    eventid_occurrences['split'] = eventid_occurrences.index.map(assign_split)
    # 4. Map back to all samples
    sampleid_to_split = dict(zip(eventid_occurrences['sample_id'], eventid_occurrences['split']))
    df_long['split'] = df_long['sample_id'].map(sampleid_to_split)
    # 5. Get indices for each split
    train_mask = df_long['split'] == 'train'
    val_mask = df_long['split'] == 'val'
    test_mask = df_long['split'] == 'test'
    # 6. Save split indices (row indices in df_long)
    pd.DataFrame({'index': df_long.index[train_mask]}).to_csv(os.path.join(output_dir, 'train_indices.csv'), index=False)
    pd.DataFrame({'index': df_long.index[val_mask]}).to_csv(os.path.join(output_dir, 'val_indices.csv'), index=False)
    pd.DataFrame({'index': df_long.index[test_mask]}).to_csv(os.path.join(output_dir, 'test_indices.csv'), index=False)
    # 7. Save sample-eventid to split mapping
    eventid_occurrences.to_csv(os.path.join(output_dir, 'sample_eventid_splits.csv'), index=False)

    # --- IMPUTATION & COLUMN DROPPING (MATCH ITERATION 6) ---
    # 1. Drop columns with >=50% missing values
    missing_frac = df_long.isna().mean()
    cols_to_drop = missing_frac[missing_frac >= 0.5].index.tolist()
    df_long = df_long.drop(columns=cols_to_drop)
    # Log dropped columns
    pd.DataFrame({'dropped_column': cols_to_drop}).to_csv(os.path.join(output_dir, 'dropped_columns_missing.csv'), index=False)

    # 2. Impute missing values
    imputation_log = []
    for col in df_long.columns:
        if col in ["sample_id", "bin_uri", "occurrences", "rel_abundance", "split"]:
            continue
        # Ensure col is a Series
        if not isinstance(df_long[col], pd.Series):
            continue
        before_na = int(df_long[col].isna().sum())
        if before_na == 0:
            continue
        if pd.api.types.is_numeric_dtype(df_long[col]):
            median_value = df_long[col].median()
            df_long[col] = df_long[col].fillna(median_value)
            imputation_log.append({'feature': col, 'method': f'median ({median_value})', 'missing_before': before_na, 'missing_after': int(df_long[col].isna().sum())})
        else:
            mode_value = df_long[col].mode().iloc[0] if len(df_long[col].mode()) > 0 else 'Unknown'
            df_long[col] = df_long[col].fillna(mode_value)
            imputation_log.append({'feature': col, 'method': f'mode ({mode_value})', 'missing_before': before_na, 'missing_after': int(df_long[col].isna().sum())})
    pd.DataFrame(imputation_log).to_csv(os.path.join(output_dir, 'imputation_log.csv'), index=False)

    # Normalize numeric features based on training set statistics, log scaling
    scaling_log = []
    for col in feature_cols_present:
        train_mean = df_long.loc[train_mask, col].mean()
        train_std = df_long.loc[train_mask, col].std(ddof=0) + 1e-10
        df_long[col] = (df_long[col] - train_mean) / train_std
        scaling_log.append({"feature": col, "mean": train_mean, "std": train_std})
    pd.DataFrame(scaling_log).to_csv(os.path.join(output_dir, "scaling_analysis.csv"), index=False)

    # Save final feature list
    pd.DataFrame({"Feature": all_feature_cols}).to_csv(os.path.join(output_dir, "final_feature_list.csv"), index=False)


    # Get train, val, test data using new split
    X_train = df_long.loc[train_mask, ["sample_id", "bin_uri"] + all_feature_cols].copy()
    y_train = df_long.loc[train_mask, ["rel_abundance"]].copy()
    X_val = df_long.loc[val_mask, ["sample_id", "bin_uri"] + all_feature_cols].copy()
    y_val = df_long.loc[val_mask, ["rel_abundance"]].copy()
    X_test = df_long.loc[test_mask, ["sample_id", "bin_uri"] + all_feature_cols].copy()
    y_test = df_long.loc[test_mask, ["rel_abundance"]].copy()

    print(f"\nObservation counts:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")

    # Check zero inflation
    zero_train = (y_train["rel_abundance"] == 0).sum()
    zero_val = (y_val["rel_abundance"] == 0).sum()
    zero_test = (y_test["rel_abundance"] == 0).sum()
    print(f"\nZero inflation:")
    print(f"  Train: {zero_train}/{len(y_train)} ({zero_train/len(y_train)*100:.1f}%)")
    print(f"  Val: {zero_val}/{len(y_val)} ({zero_val/len(y_val)*100:.1f}%)")
    print(f"  Test: {zero_test}/{len(y_test)} ({zero_test/len(y_test)*100:.1f}%)")

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"\nSaved preprocessed data to: {output_dir}/")
    print("  X_train.csv, X_val.csv, X_test.csv")
    print("  y_train.csv, y_val.csv, y_test.csv")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for Random Forest")
    parser.add_argument("--input", type=str, default="../../data/ecuador_training_data.csv",
                        help="Path to raw data file")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory for preprocessed data")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    preprocess_and_save(
        input_path=args.input,
        output_dir=args.output,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        random_state=args.seed
    )
