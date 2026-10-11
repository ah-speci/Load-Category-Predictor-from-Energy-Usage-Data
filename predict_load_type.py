from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "load_type_model.joblib"
RENAME = {
    "Lagging_Current_Reactive.Power_kVarh": "Lag_kVarh",
    "Leading_Current_Reactive_Power_kVarh": "Lead_kVarh",
    "CO2(tCO2)": "CO2",
    "Lagging_Current_Power_Factor": "Lag_PF",
    "Leading_Current_Power_Factor": "Lead_PF",
}
VALUE_COLS = ["Usage_kWh", "Lag_kVarh", "Lead_kVarh", "CO2", "Lag_PF", "Lead_PF"]
RAW_REQUIRED = {"Date_Time", *VALUE_COLS}


def spurious(series, decimals=2):
    x = series * (10 ** decimals)
    return ((x - np.round(x)).abs() > 1e-5) & series.notna()


def pf_from(usage, reactive):
    den = np.sqrt(usage**2 + reactive**2)
    return (100 * usage / den).mask(den == 0, 0.0)


def repair_readings(data, min_q=1.0, max_pf=99.0):
    out = data.copy()
    for side in ("Lag", "Lead"):
        q, pf = f"{side}_kVarh", f"{side}_PF"
        can = out["Usage_kWh"].isna() & out[q].notna() & out[pf].notna()
        zero = can & (out[pf] == 0) & (out[q] > 0)
        out.loc[zero, "Usage_kWh"] = 0.0
        mask = can & ~zero & (out[q] >= min_q) & (out[pf] > 0) & (out[pf] <= max_pf)
        out.loc[mask, "Usage_kWh"] = (
            out.loc[mask, q] * out.loc[mask, pf]
            / np.sqrt(100.0**2 - out.loc[mask, pf]**2)
        ).round(2)
    for side in ("Lag", "Lead"):
        q, pf = f"{side}_kVarh", f"{side}_PF"
        mask = out[pf].isna() & out["Usage_kWh"].notna() & out[q].notna()
        out.loc[mask, pf] = pf_from(out.loc[mask, "Usage_kWh"], out.loc[mask, q]).round(2)
        mask = (out[q].isna() & out["Usage_kWh"].notna() & (out["Usage_kWh"] > 0)
                & out[pf].notna() & (out[pf] > 0) & (out[pf] <= max_pf))
        out.loc[mask, q] = (
            out.loc[mask, "Usage_kWh"]
            * np.sqrt((100.0 / out.loc[mask, pf])**2 - 1)
        ).round(2)
    return out


def engineer_raw_features(frame):
    raw = frame.rename(columns=RENAME).copy()
    missing = RAW_REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {sorted(missing)}")
    raw["Date_Time"] = pd.to_datetime(raw["Date_Time"], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["Date_Time"])
    raw["_n_missing"] = raw.drop(columns="Date_Time").isna().sum(axis=1)
    data = (raw.sort_values(["Date_Time", "_n_missing"])
                .drop_duplicates("Date_Time", keep="first")
                .drop(columns="_n_missing").set_index("Date_Time").sort_index())
    data["NSM"] = (data.index.hour * 3600 + data.index.minute * 60).astype(float)
    for col in VALUE_COLS:
        data.loc[spurious(data[col]), col] = np.nan
    for col in ["Lag_PF", "Lead_PF"]:
        data.loc[~data[col].between(0, 100), col] = np.nan
    data = repair_readings(data)

    features = pd.DataFrame(index=data.index)
    features["minute_of_day"] = data.index.hour * 60 + data.index.minute
    features["is_winter"] = ((data.index.month >= 11) | (data.index.month <= 2)).astype(int)
    features["weekday"] = data.index.dayofweek
    features["is_weekend"] = (features["weekday"] >= 5).astype(int)
    daytime = data["Usage_kWh"].where((data.index.hour >= 10) & (data.index.hour <= 22))
    by_day = daytime.groupby(data.index.normalize())
    features["day_med"] = by_day.transform("median")
    features["day_p90"] = by_day.transform(lambda s: s.quantile(0.9))
    return features, data


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python predict_load_type.py input.csv [predictions.csv]\n"
            "Input may be the original load_data.csv or an engineered-features CSV."
        )
    input_path = Path(sys.argv[1]).expanduser()
    output_path = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else ROOT / "predictions.csv"
    bundle = joblib.load(MODEL_PATH)
    frame = pd.read_csv(input_path)

    normalized_columns = set(frame.rename(columns=RENAME).columns)
    if RAW_REQUIRED.issubset(normalized_columns):
        features, cleaned = engineer_raw_features(frame)
        result = cleaned.reset_index().rename(columns={"index": "Date_Time"})
        result = result.merge(features, left_on="Date_Time", right_index=True, how="left")
        original_labels = frame[["Date_Time", "Load_Type"]] if "Load_Type" in frame.columns else None
        if original_labels is not None:
            original_labels = original_labels.copy()
            original_labels["Date_Time"] = pd.to_datetime(original_labels["Date_Time"], dayfirst=True, errors="coerce")
            result = result.merge(original_labels.dropna(subset=["Date_Time"]).drop_duplicates("Date_Time"), on="Date_Time", how="left", suffixes=("", "_Actual"))
        X = features
    else:
        missing = [col for col in bundle["feature_columns"] if col not in frame.columns]
        if missing:
            raise ValueError(f"Input is neither a recognized raw dataset nor an engineered-features CSV. Missing feature columns: {missing}")
        result = frame.copy()
        X = frame[bundle["feature_columns"]]

    pred = bundle["estimator"].predict(X[bundle["feature_columns"]])
    if len(result) != len(pred):
        # Raw-data path uses deduplicated timestamps, so align predictions by timestamp.
        result["Predicted_Load_Type"] = pd.Series(pred, index=X.index).reindex(pd.to_datetime(result["Date_Time"], errors="coerce")).to_numpy()
    else:
        result["Predicted_Load_Type"] = pred
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Model: {bundle['model_name']} / {bundle['feature_set']}")
    if bundle.get("retrospective_features"):
        print("Note: this model uses same-day activity features; it is retrospective, not real-time.")
    print(f"Saved {len(result)} predictions to: {output_path.resolve()}")


if __name__ == "__main__":
    main()