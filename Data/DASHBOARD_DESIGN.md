# Gas Turbine HMI - Current Dashboard Design

Last updated: 2026-02-23

## Overview

This dashboard is a condition-monitoring HMI for a marine gas turbine digital twin.

It combines:

- current operating-state visualization
- model-predicted component decay
- predicted-vs-actual error inspection
- 3D degradation/fuel surface exploration
- linear remaining useful life (RUL) projections
- chat assistant support for non-technical explanations

## Current Prediction Pipeline

1. Load one snapshot from the 20% holdout split (not used for model training).
2. Read sensor/control features from that row.
3. Predict:
   - compressor decay using **SVR (RBF)**
   - turbine decay using **Random Forest**
4. Compare predicted values with actual decay labels from the same holdout row.
5. Generate:
   - health classification (`healthy`, `warning`, `critical`)
   - maintenance recommendation
   - RUL projections in dataset units

## Dashboard Sections

### 1) Action / Priority / Maintenance Card

- Displays action and priority using aligned labels:
  - `healthy`
  - `warning`
  - `critical`
- Shows:
  - **Predicted Time to Maintenance Compressor** (`N units`)
  - **Predicted Time to Maintenance Turbine** (`N units`)

### 2) Turbine 360 + Sensor Strip

- Local M501J frame sequence (manual slider control, no auto-spin)
- Operator sensors grouped by:
  - temperatures
  - pressures
  - RPM/torque
  - fuel flow

### 3) Ship Status Card

- ship speed
- lever position
- propeller torques
- TIC (control signal, not mass flow)

### 4) Health + Error Card

- Current compressor and turbine health percentages
- Predicted vs actual decay and signed error for both components
- Source metadata (holdout snapshot index)

### 5) 3D Degradation Surface

- Axes:
  - X: turbine decay (Kmt)
  - Y: compressor decay (Kmc)
  - Z: fuel flow (kg/s)
- Color encodes T48
- Hover gives exact local values
- Current state marker is overlaid

### 6) Linear RUL Projections

- Separate charts for compressor and turbine
- Each chart shows:
  - current decay point
  - threshold line
  - projected failure crossing point
  - `RUL ≈ N units`

RUL units are dataset time-index units (not direct clock hours).

## Health Classification Bands

### Compressor

- Healthy: `>= 0.98`
- Warning: `0.96 - 0.98`
- Critical: `< 0.96`

### Turbine

- Healthy: `>= 0.99`
- Warning: `0.98 - 0.99`
- Critical: `< 0.98`

System-level class is the worst of the two components.

## Data Semantics

- Snapshot values are simulated from a holdout CSV row, not live vessel telemetry.
- TIC is a turbine injection control command (%).
- Fuel_Flow is actual fuel rate (kg/s).

## Notes for Presentation

- This is condition monitoring with model-based projection.
- It is not true timestamp-based long-horizon calendar forecasting.
- Time-to-maintenance is shown as dataset units for transparent comparison across conditions.
