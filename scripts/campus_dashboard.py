import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from datetime import datetime

# ---------------------------- PATH CONFIG ---------------------------- #

BASE_DATA = Path("data")
BASE_OUTPUT = Path("output")
BASE_OUTPUT.mkdir(exist_ok=True)

IMG_DASHBOARD = BASE_OUTPUT / "dashboard.png"
CSV_CLEAN = BASE_OUTPUT / "cleaned_energy_data.csv"
CSV_SUMMARY = BASE_OUTPUT / "building_summary.csv"
TXT_SUMMARY = BASE_OUTPUT / "summary.txt"

# ---------------------------- LOGGING SETUP ---------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("dashboard.log"),
        logging.StreamHandler()
    ]
)

# ---------------------------- DATA OBJECTS ---------------------------- #

class EnergySample:
    def __init__(self, ts, value):
        self.time = pd.to_datetime(ts)
        self.value = float(value)


class Facility:
    def __init__(self, label: str):
        self.label = label
        self.samples: list[EnergySample] = []

    def insert(self, sample: EnergySample):
        self.samples.append(sample)

    def to_frame(self):
        if not self.samples:
            return pd.DataFrame(columns=["timestamp", "kwh", "building"])
        df = pd.DataFrame({
            "timestamp": [s.time for s in self.samples],
            "kwh": [s.value for s in self.samples]
        })
        df["building"] = self.label
        return df

    def daily_peak_hour(self):
        if not self.samples:
            return None, 0

        df = self.to_frame().set_index("timestamp")
        hourly = df.groupby(df.index.hour)["kwh"].sum()

        hour = int(hourly.idxmax())
        amount = float(hourly.max())
        return hour, amount


class FacilityCollection:
    def __init__(self):
        self.map = {}

    def import_csv_dir(self, folder: Path):
        csvs = list(folder.glob("*.csv"))
        if not csvs:
            logging.warning(f"No CSV files found in directory: {folder.resolve()}")

        for file in csvs:
            try:
                logging.info(f"Processing input file → {file.name}")

                df = pd.read_csv(file, parse_dates=["timestamp"], on_bad_lines="skip")
                if not {"timestamp", "kwh"}.issubset(df.columns):
                    logging.error(f"Missing required columns in {file}. Skipped.")
                    continue

                label = file.stem.replace("_usage", "").replace("_", " ").title()
                facility = Facility(label)

                df = df.dropna(subset=["timestamp", "kwh"])
                df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
                df = df.dropna(subset=["kwh"])

                for row in df.itertuples(index=False):
                    facility.insert(EnergySample(row.timestamp, row.kwh))

                self.map[label] = facility

            except Exception as exc:
                logging.exception(f"Error reading {file}: {exc}")

    def merge_as_frame(self):
        frames = [f.to_frame() for f in self.map.values()]
        if not frames:
            return pd.DataFrame(columns=["timestamp", "kwh", "building"])

        merged = pd.concat(frames, ignore_index=True)
        merged["timestamp"] = pd.to_datetime(merged["timestamp"])
        return merged.sort_values("timestamp").reset_index(drop=True)

# ---------------------------- AGGREGATE FUNCTIONS ---------------------------- #

def daily_stats(df):
    temp = df.set_index("timestamp")
    return temp.groupby("building").resample("D")["kwh"].sum().reset_index()


def weekly_stats(df):
    temp = df.set_index("timestamp")
    return temp.groupby("building").resample("W")["kwh"].sum().reset_index()


def summarize_buildings(df):
    return (
        df.groupby("building")
        .agg(
            total_kwh=("kwh", "sum"),
            mean_kwh=("kwh", "mean"),
            min_kwh=("kwh", "min"),
            max_kwh=("kwh", "max")
        )
        .reset_index()
    )

# ---------------------------- VISUALIZATION ---------------------------- #

def draw_dashboard(all_df, daily_df, weekly_df, summary_df, manager):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(3, 1, figsize=(13, 15), gridspec_kw={"height_ratios": [1, 1, 1]})

    # --- Chart 1: Rolling Daily Usage --- #
    ax0 = ax[0]
    for name, grp in all_df.groupby("building"):
        roll = grp.set_index("timestamp").resample("D")["kwh"].sum().rolling(7, min_periods=1).mean()
        ax0.plot(roll.index, roll.values, label=name)
    ax0.set_title("7-Day Rolling Energy Usage")
    ax0.legend()

    # --- Chart 2: Weekly Averages --- #
    weekly_avg = weekly_df.groupby("building")["kwh"].mean().sort_values(ascending=False)
    weekly_avg.plot(kind="bar", ax=ax[1])
    ax[1].set_title("Average Weekly Consumption")

    # --- Chart 3: Peak Hour Scatter --- #
    peak_info = []
    for name in summary_df["building"]:
        facility = manager.map[name]
        hr, kw = facility.daily_peak_hour()
        peak_info.append((name, hr, kw))

    peak_df = pd.DataFrame(peak_info, columns=["building", "hour", "kwh"])
    ax[2].scatter(peak_df["hour"], peak_df["kwh"], s=90)
    for _, row in peak_df.iterrows():
        ax[2].annotate(row["building"], (row["hour"] + 0.1, row["kwh"]))

    ax[2].set_xlabel("Hour of Day")
    ax[2].set_title("Peak Hour Energy Distribution")

    plt.tight_layout()
    plt.savefig(IMG_DASHBOARD)
    plt.close(fig)
    logging.info(f"Dashboard created → {IMG_DASHBOARD}")

# ---------------------------- OUTPUT WRITING ---------------------------- #

def save_all_outputs(clean_df, summary_df):
    clean_df.to_csv(CSV_CLEAN, index=False)
    summary_df.to_csv(CSV_SUMMARY, index=False)
    logging.info("Clean CSV and summary CSV saved.")


def write_summary_text(clean_df, summary_df):
    campus_total = clean_df["kwh"].sum()
    top_row = summary_df.loc[summary_df["total_kwh"].idxmax()]

    peak_row = clean_df.loc[clean_df["kwh"].idxmax()]
    peak_time = pd.to_datetime(peak_row["timestamp"])

    lines = [
        f"Campus Energy Report — {datetime.now().isoformat()}",
        f"Total Energy Consumption: {campus_total:.2f} kWh",
        f"Highest Consumer: {top_row['building']} ({top_row['total_kwh']:.2f} kWh)",
        f"Largest Individual Reading: {peak_row['kwh']:.2f} kWh at {peak_time} — {peak_row['building']}",
        "",
        "Detailed files saved in the output/ directory."
    ]

    TXT_SUMMARY.write_text("\n".join(lines))
    logging.info(f"Summary text written → {TXT_SUMMARY}")

# ---------------------------- MAIN PROGRAM ---------------------------- #

def main():
    logging.info("===== Energy Dashboard Pipeline Started =====")

    mgr = FacilityCollection()
    mgr.import_csv_dir(BASE_DATA)

    combined = mgr.merge_as_frame()
    if combined.empty:
        logging.error("No usable data detected; stopping.")
        return

    combined = combined.dropna(subset=["timestamp", "kwh"])
    combined["kwh"] = pd.to_numeric(combined["kwh"], errors="coerce")
    combined = combined.dropna(subset=["kwh"])
    combined = combined[combined["kwh"] >= 0].reset_index()

    daily_df = daily_stats(combined)
    weekly_df = weekly_stats(combined)
    summary_df = summarize_buildings(combined)

    draw_dashboard(combined, daily_df, weekly_df, summary_df, mgr)

    save_all_outputs(combined, summary_df)
    write_summary_text(combined, summary_df)

    logging.info("===== Pipeline Complete =====")


if __name__ == "__main__":
    main()
