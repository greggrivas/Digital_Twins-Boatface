# Gas Turbine Predictive Maintenance Dashboard

## Overview

This dashboard provides real-time health monitoring for marine vessel gas turbine propulsion systems. It uses trained Random Forest models to predict component decay from live sensor readings, enabling predictive maintenance before failures occur.

---

## How It Works

### The Core Concept

Each moment in time, the engine produces sensor readings. These readings form a "snapshot" of the engine state:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENGINE SNAPSHOT                              │
├─────────────────────────────────────────────────────────────────┤
│  MEASURABLE (from sensors)          │  HIDDEN (internal wear)  │
│  ─────────────────────────────────  │  ─────────────────────── │
│  • Ship Speed (knots)               │  • Compressor Decay      │
│  • Lever Position                   │  • Turbine Decay         │
│  • GT Shaft Torque (kN·m)           │                          │
│  • GT RPM                           │  (Can't measure these    │
│  • Gas Generator RPM                │   directly - they        │
│  • Temperatures (T48, T2)           │   represent internal     │
│  • Pressures (P48, P2, Pexh)        │   component wear)        │
│  • Fuel Flow (kg/s)                 │                          │
│  • Turbine Injection Control        │                          │
└─────────────────────────────────────────────────────────────────┘
```

### The Prediction Pipeline

```
LIVE SENSORS                    ML MODELS                      DASHBOARD
─────────────                   ─────────                      ─────────

Ship Speed ─────┐
Lever Pos ──────┤                                              ┌──────────────┐
GT Torque ──────┤              ┌─────────────────┐             │ COMPRESSOR   │
GT RPM ─────────┤              │ Random Forest   │             │ Health: 97.5%│
GG RPM ─────────┼─────────────►│ (Compressor)    │────────────►│ Status: OK   │
T48 ────────────┤              └─────────────────┘             └──────────────┘
T2 ─────────────┤
P48 ────────────┤              ┌─────────────────┐             ┌──────────────┐
P2 ─────────────┤              │ Random Forest   │             │ TURBINE      │
Pexh ───────────┤─────────────►│ (Turbine)       │────────────►│ Health: 98.8%│
TIC ────────────┤              └─────────────────┘             │ Status: OK   │
Fuel Flow ──────┘                                              └──────────────┘
(14 features)                  (trained on                     (predicted
                                11,934 samples)                 decay values)
```

---

## Dashboard Sections

### 1. Operating State Panel

Live sensor readings showing current engine operation.

| Metric | Sensor | Unit | Typical Range |
|--------|--------|------|---------------|
| Ship Speed | `Ship_Speed` | knots | 3 - 27 |
| Lever Position | `Lever_Pos` | - | 1.1 - 9.3 |
| GT Shaft Torque | `GT_Torque` | kN·m | 254 - 72,785 |
| GT RPM | `GT_RPM` | rpm | 1,308 - 3,561 |
| Gas Generator RPM | `GG_RPM` | rpm | 6,589 - 9,797 |
| Fuel Flow | `Fuel_Flow` | kg/s | 0.06 - 1.75 |

### 2. Temperature Panel

Critical temperature readings for thermal monitoring.

| Metric | Sensor | Unit | Typical Range |
|--------|--------|------|---------------|
| HP Turbine Exit Temp | `T48` | °C | 442 - 1,116 |
| Compressor Outlet Temp | `T2` | °C | 540 - 789 |

### 3. Pressure Panel

Pressure readings across the turbine system.

| Metric | Sensor | Unit | Typical Range |
|--------|--------|------|---------------|
| HP Turbine Exit Pressure | `P48` | bar | 1.09 - 4.56 |
| Compressor Outlet Pressure | `P2` | bar | 5.0 - 22.5 |
| Exhaust Pressure | `Pexh` | bar | 1.02 - 1.05 |

### 4. Health Predictions Panel

**These are the key outputs** - predicted decay coefficients from the ML models.

| Component | Predicted By | Range | Health Bands |
|-----------|--------------|-------|--------------|
| **Compressor Decay** | `rf_compressor.predict()` | 0.95 - 1.0 | |
| | | | 🟢 Healthy: ≥ 0.98 |
| | | | 🟡 Warning: 0.96 - 0.98 |
| | | | 🔴 Critical: < 0.96 |
| **Turbine Decay** | `rf_turbine.predict()` | 0.975 - 1.0 | |
| | | | 🟢 Healthy: ≥ 0.99 |
| | | | 🟡 Warning: 0.98 - 0.99 |
| | | | 🔴 Critical: < 0.98 |

> **Note:** A decay coefficient of 1.0 means a brand new component. As it decreases, the component is wearing out and losing efficiency.

### 5. 3D Visualization

Interactive surface plot showing:
- **X-axis:** Turbine Decay (0.975 - 1.0)
- **Y-axis:** Compressor Decay (0.95 - 1.0)
- **Z-axis:** Fuel Flow (kg/s)
- **Color:** Turbine Exit Temperature (T48)
- **Green Dot:** Current engine state

This visualization answers: *"Where is my engine on the degradation surface?"*

---

## Example Dashboard Reading

For a random engine snapshot:

```
╔══════════════════════════════════════════════════════════════════╗
║              GAS TURBINE DIGITAL TWIN DASHBOARD                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  OPERATING STATE                    HEALTH STATUS                ║
║  ────────────────                   ─────────────                ║
║  Ship Speed:     15.0 knots         ┌────────────────────────┐   ║
║  Lever Position: 5.5                │ COMPRESSOR  [████░] 97%│   ║
║  GT Torque:      28,450 kN·m        │ Status: 🟡 WARNING     │   ║
║  GT RPM:         2,156 rpm          └────────────────────────┘   ║
║  GG RPM:         8,234 rpm          ┌────────────────────────┐   ║
║  Fuel Flow:      0.504 kg/s         │ TURBINE     [█████] 99%│   ║
║                                     │ Status: 🟢 HEALTHY     │   ║
║  TEMPERATURES         PRESSURES     └────────────────────────┘   ║
║  ────────────         ─────────                                  ║
║  T48: 712°C           P48: 2.8 bar  RECOMMENDATION:              ║
║  T2:  645°C           P2:  14.2 bar Schedule compressor          ║
║                       Pexh: 1.03 bar inspection within 500 hrs   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Data Flow for Real Deployment

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   VESSEL    │    │    DATA     │    │     ML      │    │  DASHBOARD  │
│   SENSORS   │───►│ ACQUISITION │───►│   MODELS    │───►│     UI      │
│             │    │   SYSTEM    │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     │                   │                  │                  │
     │                   │                  │                  │
  Physical           Collect &          Predict            Display
  measurements       validate           decay              health
  from engine        sensor data        coefficients       status
```

### In Development (Current)

We simulate live data by picking random rows from the CSV:
```python
sample_idx = np.random.randint(0, len(df))
live_sensor_data = X_all.iloc[[sample_idx]]
```

### In Production

Replace with actual sensor feed:
```python
live_sensor_data = pd.DataFrame([get_live_sensor_readings()])
# or
live_sensor_data = read_from_opc_ua_server()
# or
live_sensor_data = kafka_consumer.get_latest()
```

---

## Model Performance

The Random Forest models achieve excellent accuracy:

| Target | R² Score | MAE | Cross-Validation |
|--------|----------|-----|------------------|
| Compressor Decay | 0.996 | 0.0005 | 0.996 ± 0.001 |
| Turbine Decay | 0.993 | 0.0003 | 0.992 ± 0.001 |

This means the predicted decay values are highly reliable for maintenance decisions.

---

## Files

| File | Description |
|------|-------------|
| `3dview.py` | 3D visualization with current state marker |
| `cleaned_data.csv` | 11,934 engine snapshots |
| `models/models.py` | Model training and evaluation |

---

## Future Enhancements

1. **Real-time streaming** - Connect to vessel data bus (OPC-UA, MQTT)
2. **Trend monitoring** - Track decay over time, not just current state
3. **Alerting** - Push notifications when entering warning/critical zones
4. **Historical playback** - Review past engine states
5. **What-if analysis** - Predict decay at different operating conditions
