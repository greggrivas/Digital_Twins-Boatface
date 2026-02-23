# Digital Twin for Marine Vessel Propulsion System: A Data-Driven Approach to Predictive Maintenance

**AIS4004 Digital Twins - Data**
**Group Portfolio Report**

---

## Table of Contents

1. **Introduction**
   - 1.1 Background and Motivation
   - 1.2 Problem Definition
   - 1.3 Objectives
   - 1.4 Dataset Overview
   - 1.5 Report Structure

2. **Methodology**
   - 2.1 Data Cleaning and Preprocessing
   - 2.2 Exploratory Data Analysis
   - 2.3 Machine Learning Models
   - 2.4 Model Evaluation and Comparison

3. **Digital Twin Development**
   - 3.1 Dashboard Layout and Design
   - 3.2 Metrics Selection and Justification
   - 3.3 Implementation

4. **AI-Agent Implementation**
   - 4.1 Architecture Overview
   - 4.2 Capabilities and Use Cases
   - 4.3 Example Interactions

5. **Conclusion**
   - 5.1 Summary
   - 5.2 Future Work

6. **References**

---

## 1. Introduction

### 1.1 Background and Motivation

Gas turbines serve as the primary propulsion system in modern marine vessels, offering high power output, compact design, and rapid acceleration capabilities. These advantages make them particularly valuable in military and high-speed commercial vessels. However, the harsh marine environment—characterized by salt air, humidity, and continuous high-load operation—accelerates component degradation, leading to reduced efficiency, increased fuel consumption, and potential system failures.

Traditional maintenance strategies fall into two categories: **reactive maintenance** (fixing components after failure) and **scheduled maintenance** (replacing components at fixed intervals regardless of condition). Both approaches have significant drawbacks. Reactive maintenance risks catastrophic failures at sea, while scheduled maintenance often replaces functional components prematurely, wasting resources and operational time.

**Predictive maintenance** offers a third approach: using sensor data and machine learning to predict component degradation before failure occurs. This enables maintenance to be scheduled precisely when needed—not too early, not too late. The integration of predictive models with real-time monitoring systems creates what is known as a **Digital Twin**: a virtual representation of the physical system that continuously mirrors its state and predicts its future behavior.

### 1.2 Problem Definition

This project addresses the challenge of predicting component decay in marine vessel gas turbine propulsion systems. Specifically, we focus on two critical health indicators:

- **GT Compressor Decay State Coefficient** — Represents the degradation of the gas turbine compressor (range: 0.95–1.0, where 1.0 indicates a new component)
- **GT Turbine Decay State Coefficient** — Represents the degradation of the gas turbine itself (range: 0.975–1.0)

A drop of 5–10% in efficiency is typically considered critical for gas turbines, making early detection of decay essential for operational safety and cost optimization.

**Research Questions:**
1. Which sensor parameters are most indicative of component degradation?
2. Which machine learning algorithms best predict decay coefficients from operational sensor data?
3. How can predictive models be integrated into a digital twin for real-time monitoring?
4. How can an AI agent assist operators in retrieving insights from the system?

### 1.3 Objectives

The objectives of this project align with the data science workflow:

| Phase | Objective |
|-------|-----------|
| **Define Problem** | Establish predictive maintenance goals for marine propulsion systems |
| **Collect Data** | Utilize the UCI Naval Propulsion Plants dataset (11,934 records, 18 features) |
| **Clean Data** | Handle missing values, duplicates, and ensure data consistency |
| **Explore/Visualize** | Identify patterns, correlations, and key predictive features |
| **Build Model** | Develop and compare multiple ML algorithms for decay prediction |
| **Evaluate Model** | Assess performance using MAE, MSE, and R² metrics |
| **Deploy/Monitor** | Integrate models into a digital twin dashboard with AI agent capabilities |

### 1.4 Dataset Overview

The dataset contains records from a naval vessel's gas turbine propulsion plant, designed for condition-based maintenance research. It comprises 16-dimensional feature vectors capturing steady-state operation measurements:

**Operational Parameters:**
- Lever position (lp) — Ship's throttle setting
- Ship speed (v) — Velocity in knots

**Gas Turbine Measurements:**
- GT shaft torque (GTT) — Torque on the gas turbine shaft [kN·m]
- GT rate of revolutions (GTn) — Turbine RPM
- Gas Generator rate of revolutions (GGn) — Gas generator RPM

**Propulsion Metrics:**
- Starboard Propeller Torque (Ts) [kN]
- Port Propeller Torque (Tp) [kN]

**Temperature Sensors:**
- HP Turbine exit temperature (T48) [°C]
- GT Compressor inlet air temperature (T1) [°C] — Constant at 288K
- GT Compressor outlet air temperature (T2) [°C]

**Pressure Sensors:**
- HP Turbine exit pressure (P48) [bar]
- GT Compressor inlet air pressure (P1) [bar] — Constant at ~0.998 bar
- GT Compressor outlet air pressure (P2) [bar]
- GT exhaust gas pressure (Pexh) [bar]

**Control and Fuel:**
- Turbine Injection Control (TIC) [%]
- Fuel flow (mf) [kg/s]

**Target Variables:**
- GT Compressor decay state coefficient
- GT Turbine decay state coefficient

### 1.5 Report Structure

This report follows the data science workflow, documenting our progression from problem definition to deployment:

| Section | Content |
|---------|---------|
| **1. Introduction** | Problem context, objectives, and dataset description |
| **2. Methodology** | Data cleaning, EDA, ML model development and evaluation |
| **3. Digital Twin Development** | Dashboard design, metrics selection, implementation |
| **4. AI-Agent Implementation** | Architecture, capabilities, and example interactions |
| **5. Conclusion** | Summary and future work |

---
