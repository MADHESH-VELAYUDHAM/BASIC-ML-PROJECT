import pandas as pd


def clean_data(df):
    df = df.copy()

    df = df.dropna(how="all")  # rows clean paindran
    df = df.dropna(axis=1, how="all")  # ippo columns clean paindran

    df = df.drop_duplicates()

    # columns names
    df.columns = df.columns.str.strip().str.lower()  # unwanted space remove painum and string ha lower la convert painum

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()  # string columns ha separate paindran
        #df[col] = df[col].replace({"nan": 0, "": 0, "none": 0, "None": 0, "unknown": 0})
    return df


def catogry_sep(df, category_threshold=0.5):  # threshold value default ethu data frame data pathu athoda dtype convert paindrathu
    df = df.copy()
    date_cols, numeric_cols, category_cols = [], [], []  # 3 columns separate paindran

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            date_cols.append(col)
            continue
        non_null = series.dropna()
        if len(non_null) == 0:
            category_cols.append(col)
            continue
        try:
            converted = pd.to_datetime(non_null, errors="coerce")  # datetime conversion
            success_ratio = converted.notna().median()
            if success_ratio >= 0.9:
                df[col] = pd.to_datetime(series, errors="coerce")
                date_cols.append(col)
                continue
        except Exception:
            pass

        try:
            converted = pd.to_numeric(non_null, errors="coerce")  # numeric columns
            success_ratio = converted.notna().median()
            if success_ratio >= 0.9:
                df[col] = pd.to_numeric(series, errors="coerce")
                numeric_cols.append(col)
                continue
        except Exception:
            pass

        unimass = series.nunique(dropna=True) / max(len(non_null), 1)
        if unimass<= category_threshold:                          # remaing columns
            df[col] = series.astype("category")
        category_cols.append(col)
    return df, date_cols, numeric_cols, category_cols