from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "Lever_Pos",
    "Ship_Speed",
    "GT_Torque",
    "GT_RPM",
    "GG_RPM",
    "Prop_Torque_S",
    "Prop_Torque_P",
    "T48",
    "T2",
    "P48",
    "P2",
    "Pexh",
    "TIC",
    "Fuel_Flow",
]

TARGET_COMP = "Compressor_Decay"
TARGET_TURB = "Turbine_Decay"

INTERPRETATIONS = {
    "T48_vs_turbine_decay": "High turbine exit temperature predicts blade thermal stress.",
    "T2_vs_compressor_decay": "Higher compressor outlet temperature indicates reduced compression efficiency.",
    "P48_vs_compressor_decay": "Pressure shifts can indicate seal wear and leakage behavior.",
    "Fuel_Flow_vs_decay": "Increased fuel flow for equivalent load indicates overall inefficiency.",
    "TIC_vs_turbine_decay": "Higher turbine injection control often indicates compensation for wear.",
}

SENSOR_MEANINGS = {
    "T48": ("Elevated HP turbine exit temperature indicates turbine blade thermal stress", "turbine"),
    "T2": ("Higher compressor outlet temperature indicates compressor inefficiency", "compressor"),
    "P48": ("Abnormal HP turbine exit pressure can indicate seal wear/leakage", "compressor"),
    "Fuel_Flow": ("High fuel flow for same load indicates reduced overall efficiency", "both"),
    "TIC": ("High turbine injection control indicates compensating control action", "turbine"),
}


@dataclass
class NavalPredictor:
    df: pd.DataFrame
    compressor_model: Any  # SVR for compressor (R² = 0.998)
    turbine_model: RandomForestRegressor  # Random Forest for turbine (R² = 0.993)
    scaler: StandardScaler  # Scaler for SVR

    @classmethod
    def from_csv(cls, csv_path: Path) -> "NavalPredictor":
        df = pd.read_csv(csv_path)
        X = df[FEATURES]
        y_comp = df[TARGET_COMP]
        y_turb = df[TARGET_TURB]

        # Scale features for SVR (required for good performance)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # SVR for compressor decay (R² = 0.998 per report)
        compressor_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.001)
        compressor_model.fit(X_scaled, y_comp)

        # Random Forest for turbine decay (R² = 0.993 per report)
        turbine_model = RandomForestRegressor(n_estimators=100, random_state=42)
        turbine_model.fit(X, y_turb)

        return cls(df=df, compressor_model=compressor_model, turbine_model=turbine_model, scaler=scaler)

    @staticmethod
    def severity(compressor_decay: float, turbine_decay: float) -> str:
        if compressor_decay < 0.97 or turbine_decay < 0.985:
            return "critical"
        if compressor_decay < 0.98 or turbine_decay < 0.990:
            return "warning"
        if compressor_decay < 0.99 or turbine_decay < 0.995:
            return "caution"
        return "healthy"

    def _build_feature_row(self, ship_speed: int, lever_pos: float) -> pd.DataFrame:
        df_speed = self.df[self.df["Ship_Speed"] == ship_speed]
        if df_speed.empty:
            nearest_speed = min(self.df["Ship_Speed"].unique(), key=lambda v: abs(v - ship_speed))
            df_speed = self.df[self.df["Ship_Speed"] == nearest_speed]

        idx = (df_speed["Lever_Pos"] - lever_pos).abs().idxmin()
        row = df_speed.loc[idx, FEATURES].copy()
        row["Ship_Speed"] = ship_speed
        row["Lever_Pos"] = lever_pos
        return pd.DataFrame([row], columns=FEATURES)

    def predict_from_controls(self, ship_speed: int, lever_pos: float) -> Dict[str, float | str]:
        X_pred = self._build_feature_row(ship_speed, lever_pos)
        # SVR needs scaled features for compressor
        X_scaled = self.scaler.transform(X_pred)
        compressor = float(self.compressor_model.predict(X_scaled)[0])
        # Random Forest for turbine (no scaling needed)
        turbine = float(self.turbine_model.predict(X_pred)[0])
        return {
            "compressor_decay": round(compressor, 6),
            "turbine_decay": round(turbine, 6),
            "severity": self.severity(compressor, turbine),
            "operating_condition": f"{ship_speed} knots",
            "prediction_confidence": 0.998,  # SVR compressor confidence
        }

    def predict_from_sensors(self, sensors: Dict[str, float]) -> Dict[str, float | str]:
        X_pred = pd.DataFrame([sensors], columns=FEATURES)
        # SVR needs scaled features for compressor
        X_scaled = self.scaler.transform(X_pred)
        compressor = float(self.compressor_model.predict(X_scaled)[0])
        # Random Forest for turbine (no scaling needed)
        turbine = float(self.turbine_model.predict(X_pred)[0])
        return {
            "compressor_decay": round(compressor, 6),
            "turbine_decay": round(turbine, 6),
            "severity": self.severity(compressor, turbine),
            "operating_condition": f"{int(sensors['Ship_Speed'])} knots",
            "prediction_confidence": 0.998,  # SVR compressor confidence
        }

    def dataset_summary(self) -> Dict[str, int | float | list[int]]:
        speeds = sorted(int(v) for v in self.df["Ship_Speed"].unique())

        comp = self.df[TARGET_COMP]
        turb = self.df[TARGET_TURB]

        critical = ((comp < 0.97) | (turb < 0.985)).sum()
        warning = (((comp < 0.98) | (turb < 0.990)) & ~((comp < 0.97) | (turb < 0.985))).sum()

        grouped = self.df.groupby("Ship_Speed")[[TARGET_COMP, TARGET_TURB]].mean()
        mean_health = grouped.mean(axis=1)

        return {
            "total_operating_points": int(len(self.df)),
            "speed_range": [int(min(speeds)), int(max(speeds))],
            "unique_speeds": speeds,
            "critical_conditions": int(critical),
            "warning_conditions": int(warning),
            "avg_compressor_decay": round(float(comp.mean()), 6),
            "avg_turbine_decay": round(float(turb.mean()), 6),
            "worst_speed_condition": int(mean_health.idxmin()),
            "healthiest_speed_condition": int(mean_health.idxmax()),
        }

    def sensor_analysis(self, readings: Dict[str, float]) -> Dict:
        means = self.df[FEATURES].mean()
        stds = self.df[FEATURES].std().replace(0, np.nan)

        abnormal = []
        critical_count = 0

        for sensor, value in readings.items():
            if sensor not in FEATURES:
                continue
            mean = means[sensor]
            std = stds[sensor]
            if pd.isna(std):
                continue

            z = (value - mean) / std
            if abs(z) >= 2:
                if abs(z) >= 3:
                    critical_count += 1

                low = float(mean - 2 * std)
                high = float(mean + 2 * std)
                meaning, component = SENSOR_MEANINGS.get(
                    sensor, ("Sensor deviates materially from nominal operating band", "unknown")
                )
                corr = None
                if sensor in self.df.columns:
                    corr = float(self.df[sensor].corr(self.df[TARGET_COMP]))

                abnormal.append(
                    {
                        "sensor": sensor,
                        "value": float(value),
                        "deviation": f"{z:+.1f}σ",
                        "normal_range": [round(low, 3), round(high, 3)],
                        "physical_meaning": meaning,
                        "decay_correlation": None if corr is None else round(corr, 3),
                        "affected_component": component,
                    }
                )

        return {
            "abnormal_sensors": abnormal,
            "normal_sensors": int(len(FEATURES) - len(abnormal)),
            "critical_anomalies": int(critical_count),
        }

    def correlations(self) -> Dict:
        c = self.df.corr(numeric_only=True)

        def corr(a: str, b: str) -> float:
            return round(float(c.loc[a, b]), 3)

        return {
            "temperature_correlations": {
                "T48_vs_turbine_decay": {
                    "correlation": corr("T48", TARGET_TURB),
                    "interpretation": INTERPRETATIONS["T48_vs_turbine_decay"],
                },
                "T2_vs_compressor_decay": {
                    "correlation": corr("T2", TARGET_COMP),
                    "interpretation": INTERPRETATIONS["T2_vs_compressor_decay"],
                },
            },
            "pressure_correlations": {
                "P48_vs_compressor_decay": {
                    "correlation": corr("P48", TARGET_COMP),
                    "interpretation": INTERPRETATIONS["P48_vs_compressor_decay"],
                }
            },
            "efficiency_indicators": {
                "Fuel_Flow_vs_decay": {
                    "correlation": corr("Fuel_Flow", TARGET_COMP),
                    "interpretation": INTERPRETATIONS["Fuel_Flow_vs_decay"],
                },
                "TIC_vs_turbine_decay": {
                    "correlation": corr("TIC", TARGET_TURB),
                    "interpretation": INTERPRETATIONS["TIC_vs_turbine_decay"],
                },
            },
        }

    def compare_operating_conditions(self, speed_1: int, speed_2: int) -> Dict:
        def summarize(speed: int) -> Dict[str, float | int]:
            df_speed = self.df[self.df["Ship_Speed"] == speed]
            if df_speed.empty:
                nearest = int(min(self.df["Ship_Speed"].unique(), key=lambda v: abs(v - speed)))
                df_speed = self.df[self.df["Ship_Speed"] == nearest]
                speed = nearest

            return {
                "speed": int(speed),
                "avg_T48": round(float(df_speed["T48"].mean()), 3),
                "avg_P48": round(float(df_speed["P48"].mean()), 3),
                "avg_compressor_decay": round(float(df_speed[TARGET_COMP].mean()), 6),
                "avg_turbine_decay": round(float(df_speed[TARGET_TURB].mean()), 6),
            }

        s1 = summarize(speed_1)
        s2 = summarize(speed_2)

        comp_diff = round(float(s2["avg_compressor_decay"] - s1["avg_compressor_decay"]), 6)
        turb_diff = round(float(s2["avg_turbine_decay"] - s1["avg_turbine_decay"]), 6)

        direction = "higher" if s2["speed"] > s1["speed"] else "lower"
        recommendation = (
            f"At {s2['speed']} knots ({direction} speed), monitor T48/P48 closely and limit sustained high-load operation."
        )

        return {
            "speed_1_conditions": s1,
            "speed_2_conditions": s2,
            "degradation_difference": {
                "compressor_decay_delta": comp_diff,
                "turbine_decay_delta": turb_diff,
            },
            "root_cause": "Speed changes shift temperature/pressure load, which drives thermal/mechanical stress.",
            "recommendation": recommendation,
        }


def maintenance_recommendation(compressor_decay: float, turbine_decay: float) -> Tuple[str, str, list[str], str]:
    severity = NavalPredictor.severity(compressor_decay, turbine_decay)

    if severity == "critical":
        return (
            "immediate",
            "high",
            ["turbine blades", "compressor seals"],
            "as soon as possible",
        )
    if severity == "warning":
        return (
            "scheduled",
            "medium",
            ["compressor section", "turbine hot section"],
            "within 100 operating hours",
        )
    return (
        "monitor",
        "low",
        ["routine inspection"],
        "next planned maintenance window",
    )


def _fill_pivot_grid(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """Fill sparse pivot grids to support smooth plotting/interpolation."""
    return pivot_df.sort_index().sort_index(axis=1).interpolate(axis=0).interpolate(axis=1).bfill().ffill()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bilinear_interpolate(
    x_vals: np.ndarray, y_vals: np.ndarray, grid: np.ndarray, x: float, y: float
) -> float:
    """Bilinear interpolation on a regular grid where grid[row=y, col=x]."""
    x = _clamp(float(x), float(x_vals.min()), float(x_vals.max()))
    y = _clamp(float(y), float(y_vals.min()), float(y_vals.max()))

    ix1 = int(np.searchsorted(x_vals, x))
    iy1 = int(np.searchsorted(y_vals, y))
    ix0 = max(0, ix1 - 1)
    iy0 = max(0, iy1 - 1)
    ix1 = min(ix1, len(x_vals) - 1)
    iy1 = min(iy1, len(y_vals) - 1)

    x0, x1 = float(x_vals[ix0]), float(x_vals[ix1])
    y0, y1 = float(y_vals[iy0]), float(y_vals[iy1])

    if ix0 == ix1 and iy0 == iy1:
        return float(grid[iy0, ix0])
    if ix0 == ix1:
        q0 = float(grid[iy0, ix0])
        q1 = float(grid[iy1, ix0])
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        return float(q0 * (1 - ty) + q1 * ty)
    if iy0 == iy1:
        q0 = float(grid[iy0, ix0])
        q1 = float(grid[iy0, ix1])
        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        return float(q0 * (1 - tx) + q1 * tx)

    q11 = float(grid[iy0, ix0])
    q21 = float(grid[iy0, ix1])
    q12 = float(grid[iy1, ix0])
    q22 = float(grid[iy1, ix1])
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    return float(
        q11 * (1 - tx) * (1 - ty)
        + q21 * tx * (1 - ty)
        + q12 * (1 - tx) * ty
        + q22 * tx * ty
    )


def hmi_severity(compressor_decay: float, turbine_decay: float) -> str:
    if compressor_decay < 0.96 or turbine_decay < 0.98:
        return "critical"
    if compressor_decay < 0.98 or turbine_decay < 0.99:
        return "warning"
    return "healthy"


def _speed_subset(df: pd.DataFrame, speed: int) -> tuple[pd.DataFrame, int]:
    exact = df[df["Ship_Speed"] == speed]
    if not exact.empty:
        return exact, speed
    nearest = int(min(df["Ship_Speed"].unique(), key=lambda v: abs(v - speed)))
    return df[df["Ship_Speed"] == nearest], nearest


def _display_meta(df: pd.DataFrame) -> Dict:
    cols = {
        "ship_speed": "Ship_Speed",
        "lever_pos": "Lever_Pos",
        "gt_torque": "GT_Torque",
        "gt_rpm": "GT_RPM",
        "gg_rpm": "GG_RPM",
        "fuel_flow": "Fuel_Flow",
        "t48": "T48",
        "t2": "T2",
        "p48": "P48",
        "p2": "P2",
        "pexh": "Pexh",
    }
    ranges = {k: [float(df[v].min()), float(df[v].max())] for k, v in cols.items()}
    return {
        "units": {
            "ship_speed": "knots",
            "lever_pos": "-",
            "gt_torque": "kN·m",
            "gt_rpm": "rpm",
            "gg_rpm": "rpm",
            "fuel_flow": "kg/s",
            "t48": "°C",
            "t2": "°C",
            "p48": "bar",
            "p2": "bar",
            "pexh": "bar",
        },
        "ranges": ranges,
        "health_bands": {
            "compressor": {"healthy": ">=0.98", "warning": "0.96-0.98", "critical": "<0.96"},
            "turbine": {"healthy": ">=0.99", "warning": "0.98-0.99", "critical": "<0.98"},
        },
    }


def build_hmi_snapshot(predictor: NavalPredictor) -> Dict:
    idx = int(np.random.randint(0, len(predictor.df)))
    row = predictor.df.iloc[idx]
    features = row[FEATURES].to_dict()
    pred = predictor.predict_from_sensors(features)

    compressor = float(pred["compressor_decay"])
    turbine = float(pred["turbine_decay"])

    return {
        "snapshot_id": idx,
        "source": "csv_random_row",
        "operating_state": {
            "ship_speed": int(row["Ship_Speed"]),
            "lever_pos": round(float(row["Lever_Pos"]), 3),
            "gt_torque": round(float(row["GT_Torque"]), 3),
            "gt_rpm": round(float(row["GT_RPM"]), 3),
            "gg_rpm": round(float(row["GG_RPM"]), 3),
            "tic": round(float(row["TIC"]), 3),
            "fuel_flow": round(float(row["Fuel_Flow"]), 3),
        },
        "ship_state": {
            "prop_torque_s": round(float(row["Prop_Torque_S"]), 3),
            "prop_torque_p": round(float(row["Prop_Torque_P"]), 3),
        },
        "temperature_state": {
            "t1": round(float(row["T1"]), 3),
            "t48": round(float(row["T48"]), 3),
            "t2": round(float(row["T2"]), 3),
        },
        "pressure_state": {
            "p1": round(float(row["P1"]), 3),
            "p48": round(float(row["P48"]), 3),
            "p2": round(float(row["P2"]), 3),
            "pexh": round(float(row["Pexh"]), 3),
        },
        "predictions": {
            "compressor_decay_pred": round(compressor, 6),
            "turbine_decay_pred": round(turbine, 6),
            "compressor_decay_actual": round(float(row["Compressor_Decay"]), 6),
            "turbine_decay_actual": round(float(row["Turbine_Decay"]), 6),
            "severity": hmi_severity(compressor, turbine),
            "confidence_ref": 0.998,  # SVR for compressor (0.998), RF for turbine (0.993)
        },
        "display_meta": _display_meta(predictor.df),
    }


def build_surface_data(predictor: NavalPredictor, speed: int) -> Dict:
    df_speed, used_speed = _speed_subset(predictor.df, speed)
    fuel_pivot = _fill_pivot_grid(
        df_speed.pivot_table(values="Fuel_Flow", index=TARGET_COMP, columns=TARGET_TURB)
    )
    temp_pivot = _fill_pivot_grid(
        df_speed.pivot_table(values="T48", index=TARGET_COMP, columns=TARGET_TURB)
    )

    x_vals = fuel_pivot.columns.values.astype(float)
    y_vals = fuel_pivot.index.values.astype(float)
    fuel = fuel_pivot.values.astype(float)
    temp = temp_pivot.values.astype(float)

    return {
        "speed": int(used_speed),
        "axes": {
            "turbine_decay_values": x_vals.tolist(),
            "compressor_decay_values": y_vals.tolist(),
        },
        "surface": {
            "fuel_flow_z": fuel.tolist(),
            "t48_color": temp.tolist(),
        },
        "plot_bounds": {
            "x_min": float(x_vals.min()),
            "x_max": float(x_vals.max()),
            "y_min": float(y_vals.min()),
            "y_max": float(y_vals.max()),
            "z_min": float(fuel.min()),
            "z_max": float(fuel.max()),
        },
        "axis_meta": {
            "invert_x": True,
            "invert_y": True,
            "x_label": "Turbine Decay",
            "y_label": "Compressor Decay",
            "z_label": "Fuel Flow (kg/s)",
            "color_label": "T48 (°C)",
        },
        "defaults": {
            "camera_elev": 30,
            "camera_azim": 135,
        },
    }


def build_surface_marker(
    predictor: NavalPredictor, speed: int, compressor_decay_pred: float, turbine_decay_pred: float
) -> Dict:
    surface = build_surface_data(predictor, speed)
    x_vals = np.array(surface["axes"]["turbine_decay_values"], dtype=float)
    y_vals = np.array(surface["axes"]["compressor_decay_values"], dtype=float)
    fuel = np.array(surface["surface"]["fuel_flow_z"], dtype=float)
    temp = np.array(surface["surface"]["t48_color"], dtype=float)

    fuel_value = _bilinear_interpolate(x_vals, y_vals, fuel, turbine_decay_pred, compressor_decay_pred)
    temp_value = _bilinear_interpolate(x_vals, y_vals, temp, turbine_decay_pred, compressor_decay_pred)

    return {
        "speed": int(surface["speed"]),
        "marker": {
            "turbine_decay": round(float(turbine_decay_pred), 6),
            "compressor_decay": round(float(compressor_decay_pred), 6),
            "fuel_flow": round(float(fuel_value), 6),
            "t48": round(float(temp_value), 3),
        },
    }
