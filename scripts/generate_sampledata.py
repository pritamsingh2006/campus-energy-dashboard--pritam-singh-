import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_PNG = OUTPUT_DIR / "dashboard.png"
CLEANED_CSV = OUTPUT_DIR / "cleaned_energy_data.csv"
BUILDING_SUMMARY_CSV = OUTPUT_DIR / "building_summary.csv"
SUMMARY_TXT = OUTPUT_DIR / "summary.txt"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("dashboard.log"), logging.StreamHandler()])


class MeterReading:
    def __init__(self, timestamp: pd.Timestamp, kwh: float):
        self.timestamp = pd.to_datetime(timestamp)
        self.kwh = float(kwh)

class Building:
    def __init__(self, name: str):
        self.name = name
        self.readings = [] 

    def add_reading(self, reading: MeterReading):
        self.readings.append(reading)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.readings:
            return pd.DataFrame(columns=["timestamp", "kwh", "building"])
        df = pd.DataFrame([{"timestamp": r.timestamp, "kwh": r.kwh} for r in self.readings])
        df["building"] = self.name
        return df

    def total_consumption(self):
        return sum(r.kwh for r in self.readings)

    def avg_consumption(self):
        return np.mean([r.kwh for r in self.readings]) if self.readings else 0

    def peak_hour(self):
        if not self.readings:
            return None, 0
        df = self.to_dataframe().set_index("timestamp")
        hour_sums = df["kwh"].groupby(df.index.hour).sum()
        peak_hour = int(hour_sums.idxmax())
        return peak_hour, float(hour_sums.max())

class BuildingManager:
    def __init__(self):
        self.buildings = {}

    def load_from_csvs(self, data_dir: Path):
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            logging.warning(f"No CSV files found in {data_dir.resolve()}")
        for csv in csv_files:
            try:
                logging.info(f"Reading {csv}")
                df = pd.read_csv(csv, parse_dates=["timestamp"], dayfirst=False, on_bad_lines='skip')
                
                building_name = csv.stem.replace("_usage", "").replace("_", " ").title()
                building = Building(building_name)
                
                if "timestamp" not in df.columns or "kwh" not in df.columns:
                    logging.error(f"File {csv} missing required columns. Skipping.")
                    continue
                
                df = df.dropna(subset=["timestamp", "kwh"])
                
                df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
                df = df.dropna(subset=["kwh"])
                for _, row in df.iterrows():
                    building.add_reading(MeterReading(row["timestamp"], row["kwh"]))
                self.buildings[building_name] = building
            except FileNotFoundError:
                logging.error(f"File not found: {csv}")
            except pd.errors.EmptyDataError:
                logging.error(f"No data in file: {csv}")
            except Exception as e:
                logging.exception(f"Failed reading {csv}: {e}")

    def combined_dataframe(self):
        dfs = []
        for b in self.buildings.values():
            dfs.append(b.to_dataframe())
        if not dfs:
            return pd.DataFrame(columns=["timestamp", "kwh", "building"])
        df = pd.concat(dfs, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df


def calculate_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("timestamp")
    daily = df.groupby("building").resample("D")["kwh"].sum().reset_index()
    return daily

def calculate_weekly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("timestamp")
    weekly = df.groupby("building").resample("W")["kwh"].sum().reset_index()
    return weekly

def building_wise_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("building").agg(
        total_kwh=("kwh", "sum"),
        mean_kwh=("kwh", "mean"),
        min_kwh=("kwh", "min"),
        max_kwh=("kwh", "max")
    ).reset_index()
    return summary


def create_dashboard_plot(df: pd.DataFrame, daily: pd.DataFrame, weekly: pd.DataFrame, summary_df: pd.DataFrame):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios':[1.2, 0.9, 1]})

    
    ax = axes[0]
    for bname, group in df.groupby("building"):
        g = group.set_index("timestamp").resample("D")["kwh"].sum().rolling(7, min_periods=1).mean()
        ax.plot(g.index, g.values, label=bname)
    ax.set_title("7-day Rolling Daily Consumption — Buildings")
    ax.set_ylabel("kWh (7-day avg)")
    ax.legend(loc="upper right")

    
    ax = axes[1]
    avg_weekly = weekly.groupby("building")["kwh"].mean().sort_values(ascending=False)
    avg_weekly.plot(kind="bar", ax=ax)
    ax.set_title("Average Weekly Consumption per Building")
    ax.set_ylabel("kWh / week")

    
    ax = axes[2]
    peaks = []
    for b in summary_df['building']:
        building_obj = manager.buildings[b]
        peak_h, peak_val = building_obj.peak_hour()
        peaks.append((b, peak_h, peak_val))
    p_df = pd.DataFrame(peaks, columns=["building", "peak_hour", "peak_kwh"])
    ax.scatter(p_df["peak_hour"], p_df["peak_kwh"], s=100)
    for i, r in p_df.iterrows():
        ax.text(r["peak_hour"] + 0.1, r["peak_kwh"], r["building"], fontsize=9)
    ax.set_xlabel("Hour of day (0-23)")
    ax.set_ylabel("Total kWh during that hour (sum over dataset)")
    ax.set_title("Peak-hour consumption per building (hour vs total kWh)")

    plt.tight_layout()
    fig.savefig(DASHBOARD_PNG)
    logging.info(f"Saved dashboard image to {DASHBOARD_PNG}")
    plt.close(fig)


def save_outputs(clean_df: pd.DataFrame, summary_df: pd.DataFrame):
    clean_df.to_csv(CLEANED_CSV, index=False)
    summary_df.to_csv(BUILDING_SUMMARY_CSV, index=False)
    logging.info(f"Saved cleaned data to {CLEANED_CSV} and summary to {BUILDING_SUMMARY_CSV}")

def write_text_summary(clean_df: pd.DataFrame, summary_df: pd.DataFrame):
    total_campus = clean_df["kwh"].sum()
    highest_building = summary_df.loc[summary_df["total_kwh"].idxmax()]
    
    peak_row = clean_df.loc[clean_df["kwh"].idxmax()]
    peak_time = pd.to_datetime(peak_row["timestamp"])
    lines = [
        f"Campus Energy Summary - generated {datetime.now().isoformat()}",
        f"Total campus consumption (kWh): {total_campus:.2f}",
        f"Highest-consuming building: {highest_building['building']} with {highest_building['total_kwh']:.2f} kWh",
        f"Peak reading: {peak_row['kwh']:.2f} kWh at {peak_time} (building: {peak_row['building']})",
        "",
        "Weekly and daily trends saved in output/*.csv and dashboard image."
    ]
    REPORT_TXT = SUMMARY_TXT
    REPORT_TXT.write_text("\n".join(lines))
    logging.info(f"Wrote summary to {REPORT_TXT}")


def main():
    global manager
    manager = BuildingManager()
    logging.info("Starting data ingestion...")
    manager.load_from_csvs(DATA_DIR)

    logging.info("Combining data...")
    combined = manager.combined_dataframe()
    if combined.empty:
        logging.error("No data available after ingest. Exiting.")
        return


    combined = combined.dropna(subset=["timestamp", "kwh"])
    combined["kwh"] = pd.to_numeric(combined["kwh"], errors="coerce")
    combined = combined.dropna(subset=["kwh"])
    combined = combined[combined["kwh"] >= 0].reset_index(drop=True)

    logging.info("Calculating aggregates...")
    daily = calculate_daily_totals(combined)
    weekly = calculate_weekly_aggregates(combined)
    summary_df = building_wise_summary(combined)

    logging.info("Creating visual dashboard...")
    create_dashboard_plot(combined, daily, weekly, summary_df)

    logging.info("Saving outputs...")
    save_outputs(combined, summary_df)
    write_text_summary(combined, summary_df)

    logging.info("All done!")

if __name__ == "__main__":
    main()