import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# =========================================================
# SETUP
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR.parent / "Images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
TARGET_FOLDERS = {
    "Compressor_Decay": IMAGE_DIR / "compressor",
    "Turbine_Decay": IMAGE_DIR / "turbine",
}
for folder in TARGET_FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE_DIR / 'cleaned_data.csv')


def image_path(target, filename):
    return os.path.join(TARGET_FOLDERS[target], filename)


def table_path(filename):
    return os.path.join(TABLE_DIR, filename)


def save_for_all_targets(filename, **kwargs):
    for target in TARGET_FOLDERS:
        plt.savefig(image_path(target, filename), **kwargs)
    print(f"Saved: {filename} -> compressor/, turbine/")


def save_for_target(target, filename, **kwargs):
    plt.savefig(image_path(target, filename), **kwargs)
    print(f"Saved: {filename} -> {TARGET_FOLDERS[target].name}/")

# =========================================================
# 1. DATA UNDERSTANDING
# =========================================================
print("=" * 70)
print("1. DATA UNDERSTANDING")
print("=" * 70)

print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Column Names and Types ---")
print(df.dtypes)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

# Data info summary figure used in appendix
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
info_text = (
    f"Rows: {df.shape[0]}\n"
    f"Columns: {df.shape[1]}\n"
    f"Numeric columns: {df.select_dtypes(include=[np.number]).shape[1]}\n"
    f"Missing values: {int(df.isna().sum().sum())}\n"
    f"Duplicate rows: {int(df.duplicated().sum())}\n"
    f"Constant columns: {', '.join([c for c in df.columns if df[c].nunique() <= 1])}"
)
ax.text(0.02, 0.95, "Dataset Information", fontsize=16, fontweight='bold', va='top')
ax.text(0.02, 0.78, info_text, fontsize=12, va='top')
plt.tight_layout()
save_for_all_targets("datainfo.png", dpi=300)
plt.close()

# =========================================================
# 2. DATA QUALITY CHECKS
# =========================================================
print("\n" + "=" * 70)
print("2. DATA QUALITY CHECKS")
print("=" * 70)

# Missing values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
print(missing_df[missing_df['Missing'] > 0] if missing.sum() > 0 else "No missing values found!")

# Duplicates
duplicates = df.duplicated().sum()
print(f"\n--- Duplicate Rows ---")
print(f"Duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")

# Constant/near-constant columns
print("\n--- Constant or Near-Constant Columns ---")
for col in df.columns:
    unique_count = df[col].nunique()
    if unique_count <= 5:
        print(f"  {col}: {unique_count} unique values")

constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
print(f"\nColumns to remove (constant): {constant_cols if constant_cols else 'None'}")

# Range validation
print("\n--- Range Validation ---")
print("Checking physical bounds...")
if 'Compressor_Decay' in df.columns:
    comp_min, comp_max = df['Compressor_Decay'].min(), df['Compressor_Decay'].max()
    print(f"  Compressor_Decay: [{comp_min:.4f}, {comp_max:.4f}] (expected: 0.95-1.0)")

if 'Turbine_Decay' in df.columns:
    turb_min, turb_max = df['Turbine_Decay'].min(), df['Turbine_Decay'].max()
    print(f"  Turbine_Decay: [{turb_min:.4f}, {turb_max:.4f}] (expected: 0.975-1.0)")

if 'Ship_Speed' in df.columns:
    speed_min, speed_max = df['Ship_Speed'].min(), df['Ship_Speed'].max()
    print(f"  Ship_Speed: [{speed_min}, {speed_max}] knots")

# =========================================================
# 3. DESCRIPTIVE STATISTICS
# =========================================================
print("\n" + "=" * 70)
print("3. DESCRIPTIVE STATISTICS")
print("=" * 70)

desc_stats = df.describe().T
desc_stats['range'] = desc_stats['max'] - desc_stats['min']
desc_stats['cv'] = (desc_stats['std'] / desc_stats['mean'] * 100).round(2)  # Coefficient of variation
print(desc_stats)

# Save to CSV
desc_stats.to_csv(table_path("descriptive_statistics.csv"))
print(f"\nSaved to: {table_path('descriptive_statistics.csv')}")

# Descriptive statistics figure used in appendix
stats_cols = ['mean', 'std', 'min', 'max']
stats_view = desc_stats[stats_cols].round(4).head(12)
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
table = ax.table(
    cellText=stats_view.values,
    colLabels=stats_view.columns,
    rowLabels=stats_view.index,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.2)
ax.set_title('Descriptive Statistics (selected columns)', fontsize=12, pad=12)
plt.tight_layout()
save_for_all_targets("descr_statistics.png", dpi=300)
plt.close()

# =========================================================
# 4. TARGET VARIABLE ANALYSIS
# =========================================================
print("\n" + "=" * 70)
print("4. TARGET VARIABLE ANALYSIS")
print("=" * 70)

targets = ['Compressor_Decay', 'Turbine_Decay']
for target in targets:
    if target in df.columns:
        print(f"\n--- {target} ---")
        print(f"  Mean: {df[target].mean():.6f}")
        print(f"  Std:  {df[target].std():.6f}")
        print(f"  Min:  {df[target].min():.6f}")
        print(f"  Max:  {df[target].max():.6f}")
        print(f"  Range: {df[target].max() - df[target].min():.6f}")
        print(f"  Skewness: {df[target].skew():.4f}")
        print(f"  Kurtosis: {df[target].kurtosis():.4f}")

# Target distributions plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, target in enumerate(targets):
    if target in df.columns:
        sns.histplot(df[target], bins=30, kde=True, ax=axes[i], color=['blue', 'red'][i])
        axes[i].axvline(df[target].mean(), color='black', linestyle='--', label=f'Mean: {df[target].mean():.4f}')
        axes[i].set_title(f'{target} Distribution')
        axes[i].set_xlabel(target)
        axes[i].legend()
plt.tight_layout()
save_for_all_targets("target_distributions.png", dpi=300)
plt.close()

# =========================================================
# 5. FEATURE DISTRIBUTIONS
# =========================================================
print("\n" + "=" * 70)
print("5. FEATURE DISTRIBUTIONS")
print("=" * 70)

# Exclude targets for feature analysis
feature_cols = [col for col in df.columns if col not in targets]
n_features = len(feature_cols)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(col, fontsize=10)
    axes[i].tick_params(labelsize=8)

# Hide empty subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Feature Distributions', fontsize=14, y=1.02)
plt.tight_layout()
save_for_all_targets("feature_distributions.png", dpi=300)
plt.close()

# =========================================================
# 6. OUTLIER DETECTION
# =========================================================
print("\n" + "=" * 70)
print("6. OUTLIER DETECTION (IQR Method)")
print("=" * 70)

outlier_counts = {}
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    if outliers > 0:
        outlier_counts[col] = outliers

if outlier_counts:
    print("Columns with outliers (IQR method):")
    for col, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {col}: {count} outliers ({count/len(df)*100:.2f}%)")
else:
    print("No outliers detected using IQR method.")

# Boxplots for all numeric features
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    if df[col].dtype in [np.float64, np.int64]:
        sns.boxplot(y=df[col], ax=axes[i], color='steelblue')
        axes[i].set_title(col, fontsize=10)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Feature Boxplots (Outlier Detection)', fontsize=14, y=1.02)
plt.tight_layout()
save_for_all_targets("feature_boxplots.png", dpi=300)
plt.close()

# =========================================================
# 7. CORRELATION ANALYSIS
# =========================================================
print("\n" + "=" * 70)
print("7. CORRELATION ANALYSIS")
print("=" * 70)

corr_matrix = df.corr()

# Full correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
save_for_all_targets("correlation_heatmap.png", dpi=300)
plt.close()

# Correlation with targets
print("\n--- Correlation with Target Variables ---")
for target in targets:
    if target in df.columns:
        print(f"\n{target}:")
        target_corr = corr_matrix[target].drop(targets).sort_values(key=abs, ascending=False)
        for feat, corr in target_corr.head(5).items():
            print(f"  {feat}: {corr:.4f}")

# Correlation bar plots for each target (top absolute correlations)
for target in targets:
    if target not in df.columns:
        continue

    target_corr = corr_matrix[target].drop(targets, errors='ignore').sort_values(key=np.abs, ascending=False)
    top_corr = target_corr.head(10)

    plt.figure(figsize=(10, 5))
    colors = ['forestgreen' if val >= 0 else 'indianred' for val in top_corr.values]
    sns.barplot(x=top_corr.values, y=top_corr.index, orient='h', palette=colors)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('Pearson Correlation Coefficient')
    plt.ylabel('Feature')
    plt.title(f'Top 10 Feature Correlations with {target}')
    plt.tight_layout()
    save_for_target(target, f"target_correlation_bar_{target}.png", dpi=300)
    plt.close()

# Focused univariate plots for key engineering variables used in report narrative
key_feature_candidates = ['Fuel_Flow', 'GG_RPM', 'T48', 'T2', 'TIC']
key_features = [col for col in key_feature_candidates if col in df.columns]
if key_features:
    fig, axes = plt.subplots(len(key_features), 2, figsize=(12, 3.2 * len(key_features)))
    if len(key_features) == 1:
        axes = np.array([axes])
    for i, col in enumerate(key_features):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i, 0], color='steelblue')
        axes[i, 0].set_title(f'{col} distribution')
        sns.boxplot(y=df[col], ax=axes[i, 1], color='lightgray')
        axes[i, 1].set_title(f'{col} boxplot')
    plt.tight_layout()
    save_for_all_targets("key_feature_univariate_panel.png", dpi=300)
    plt.close()

# =========================================================
# 8. MULTICOLLINEARITY ANALYSIS
# =========================================================
print("\n" + "=" * 70)
print("8. MULTICOLLINEARITY ANALYSIS")
print("=" * 70)

# Find highly correlated feature pairs
high_corr_pairs = []
feature_only_corr = df.drop(columns=targets, errors='ignore').corr()

for i in range(len(feature_only_corr.columns)):
    for j in range(i+1, len(feature_only_corr.columns)):
        corr_val = feature_only_corr.iloc[i, j]
        if abs(corr_val) > 0.9:
            high_corr_pairs.append({
                'Feature 1': feature_only_corr.columns[i],
                'Feature 2': feature_only_corr.columns[j],
                'Correlation': corr_val
            })

if high_corr_pairs:
    print("Highly correlated feature pairs (|r| > 0.9):")
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
    print(high_corr_df.to_string(index=False))
    high_corr_df.to_csv(table_path("high_correlations.csv"), index=False)
else:
    print("No highly correlated feature pairs found.")

# VIF calculation (if statsmodels available)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Remove constant columns first
    X_vif = df.drop(columns=targets + ['T1', 'P1'], errors='ignore')
    X_vif = X_vif.select_dtypes(include=[np.number])

    vif_data = pd.DataFrame()
    vif_data['Feature'] = X_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
    vif_data = vif_data.sort_values('VIF', ascending=False)

    print("\n--- Variance Inflation Factor (VIF) ---")
    print("VIF > 10 indicates high multicollinearity")
    print(vif_data.to_string(index=False))
    vif_data.to_csv(table_path("vif_analysis.csv"), index=False)

except ImportError:
    print("\nstatsmodels not installed. Skipping VIF analysis.")
    print("Install with: pip install statsmodels")

# =========================================================
# 9. OPERATING REGIME ANALYSIS
# =========================================================
print("\n" + "=" * 70)
print("9. OPERATING REGIME ANALYSIS")
print("=" * 70)

# Define speed regimes
if 'Ship_Speed' in df.columns:
    speed_bins = [0, 10, 15, 20, 30]
    speed_labels = ['Low (0-10)', 'Medium (10-15)', 'High (15-20)', 'Very High (20+)']
    df['Speed_Regime'] = pd.cut(df['Ship_Speed'], bins=speed_bins, labels=speed_labels)

    print("\n--- Sample Distribution by Speed Regime ---")
    regime_counts = df['Speed_Regime'].value_counts().sort_index()
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} samples ({count/len(df)*100:.1f}%)")

    # Mean decay by regime
    print("\n--- Mean Decay by Speed Regime ---")
    regime_decay = df.groupby('Speed_Regime', observed=False)[targets].mean()
    print(regime_decay)
    regime_decay.to_csv(table_path("decay_by_regime.csv"))

    # Plot decay distributions by regime
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, target in enumerate(targets):
        if target in df.columns:
            sns.boxplot(x='Speed_Regime', y=target, data=df, ax=axes[i], color='steelblue')
            axes[i].set_title(f'{target} by Speed Regime')
            axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    save_for_all_targets("decay_by_speed_regime.png", dpi=300)
    plt.close()

# =========================================================
# 10. FEATURE-TARGET RELATIONSHIPS
# =========================================================
print("\n" + "=" * 70)
print("10. FEATURE-TARGET RELATIONSHIPS")
print("=" * 70)

# Scatter plots for top correlated features with each target
for target in targets:
    if target not in df.columns:
        continue

    # Get top 6 correlated features
    target_corr = corr_matrix[target].drop(targets, errors='ignore').abs().sort_values(ascending=False)
    top_features = target_corr.head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        sns.scatterplot(x=df[feat], y=df[target], ax=axes[i], alpha=0.3, s=10)
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel(target)
        corr_val = corr_matrix.loc[feat, target]
        axes[i].set_title(f'{feat} vs {target}\n(r = {corr_val:.3f})')

    plt.suptitle(f'Top Correlated Features with {target}', fontsize=14)
    plt.tight_layout()
    save_for_target(target, f"scatter_{target}_top_features.png", dpi=300)
    plt.close()

# =========================================================
# 11. CONTROLLED OPERATING POINT ANALYSIS
# =========================================================
print("\n" + "=" * 70)
print("11. CONTROLLED OPERATING POINT ANALYSIS")
print("=" * 70)

# Analyze decay effects at fixed speed points
if 'Ship_Speed' in df.columns:
    fixed_speeds = [9, 12, 15, 18]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, speed in enumerate(fixed_speeds):
        speed_data = df[df['Ship_Speed'] == speed]
        if len(speed_data) > 10:
            axes[i].scatter(speed_data['Compressor_Decay'], speed_data['P2'],
                          alpha=0.5, label='Compressor vs P2', color='blue')
            axes[i].set_xlabel('Compressor Decay')
            axes[i].set_ylabel('Outlet Pressure (P2)')
            axes[i].set_title(f'Decay vs Pressure at {speed} knots (n={len(speed_data)})')
            axes[i].grid(True, alpha=0.3)

    plt.suptitle('Controlled Operating Point Analysis', fontsize=14)
    plt.tight_layout()
    save_for_all_targets("controlled_operating_points.png", dpi=300)
    plt.close()

# Original 15 knots analysis
if 'Ship_Speed' in df.columns:
    speed_15 = df[df['Ship_Speed'] == 15]

    if len(speed_15) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Compressor Decay vs P2
        axes[0].scatter(speed_15['Compressor_Decay'], speed_15['P2'], alpha=0.7)
        axes[0].set_title('Compressor Decay vs Outlet Pressure (P2) at 15 knots')
        axes[0].set_xlabel('Compressor Decay (1.0 = New, 0.95 = Worn)')
        axes[0].set_ylabel('Compressor Outlet Pressure (P2) [bar]')
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # Turbine Decay vs T48
        axes[1].scatter(speed_15['Turbine_Decay'], speed_15['T48'], alpha=0.7, color='orange')
        axes[1].set_title('Turbine Decay vs Exit Temperature (T48) at 15 knots')
        axes[1].set_xlabel('Turbine Decay (1.0 = New, 0.975 = Worn)')
        axes[1].set_ylabel('HP Turbine Exit Temperature (T48) [°C]')
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        save_for_all_targets("decay_effects_15knots.png", dpi=300)
        plt.close()

# =========================================================
# 12. OPERATING LINES
# =========================================================
print("\n" + "=" * 70)
print("12. OPERATING LINES ANALYSIS")
print("=" * 70)

if 'Ship_Speed' in df.columns and 'Fuel_Flow' in df.columns:
    plt.figure(figsize=(12, 7))

    # Get representative decay states
    available_decays = sorted(df['Compressor_Decay'].unique())
    selected_decays = [available_decays[0], available_decays[len(available_decays)//2], available_decays[-1]]

    df_lines = df[df['Compressor_Decay'].isin(selected_decays)]

    sns.lineplot(
        data=df_lines,
        x='Ship_Speed',
        y='Fuel_Flow',
        hue='Compressor_Decay',
        palette='viridis',
        style='Compressor_Decay',
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=8
    )

    plt.title('Operating Lines: Fuel Flow vs. Ship Speed\n(Comparing Engine Health States)', fontsize=14)
    plt.xlabel('Ship Speed (knots)', fontsize=12)
    plt.ylabel('Fuel Flow (kg/s)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Compressor Decay State')
    plt.tight_layout()
    save_for_all_targets("operating_lines.png", dpi=300)
    plt.close()

# 3D surface plot for report
if all(col in df.columns for col in ['Ship_Speed', 'Compressor_Decay', 'Fuel_Flow']):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    surface = (
        df.groupby(['Ship_Speed', 'Compressor_Decay'], observed=False)['Fuel_Flow']
        .mean()
        .reset_index()
        .pivot(index='Compressor_Decay', columns='Ship_Speed', values='Fuel_Flow')
    )
    X_grid, Y_grid = np.meshgrid(surface.columns.values, surface.index.values)
    Z_grid = surface.values
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('Ship Speed (knots)')
    ax.set_ylabel('Compressor Decay')
    ax.set_zlabel('Fuel Flow (kg/s)')
    ax.set_title('Fuel Consumption Surface')
    fig.colorbar(surf, shrink=0.5, aspect=12, label='Fuel Flow')
    plt.tight_layout()
    save_for_all_targets("fuel_consumption_3d_surface.png", dpi=300)
    plt.close()

# =========================================================
# 13. SENSOR BOXPLOTS (GROUPED BY UNIT)
# =========================================================
print("\n" + "=" * 70)
print("13. SENSOR ANALYSIS BY CATEGORY")
print("=" * 70)

# Temperature sensors
if all(col in df.columns for col in ['T48', 'T2']):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df[['T48', 'T2']], palette="Set2")
    plt.title('Temperature Sensors (°C)')
    plt.ylabel('Temperature')
    save_for_all_targets("box_plot_temperatures.png", dpi=300)
    plt.close()

# Pressure sensors
if all(col in df.columns for col in ['P2', 'P48']):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df[['P2', 'P48']], palette="Set3")
    plt.title('Pressure Sensors (bar)')
    plt.ylabel('Pressure')
    save_for_all_targets("box_plot_pressures.png", dpi=300)
    plt.close()

# Fuel flow
if 'Fuel_Flow' in df.columns:
    plt.figure(figsize=(6, 6))
    sns.boxplot(y=df['Fuel_Flow'], color="skyblue")
    plt.title('Fuel Flow Distribution')
    plt.ylabel('kg/s')
    save_for_all_targets("box_plot_fuel.png", dpi=300)
    plt.close()

# =========================================================
# 14. PAIRPLOT (SELECTED FEATURES)
# =========================================================
print("\n" + "=" * 70)
print("14. PAIRPLOT ANALYSIS")
print("=" * 70)

# Select key features for pairplot
key_features = ['Ship_Speed', 'Fuel_Flow', 'GT_Torque', 'Compressor_Decay', 'Turbine_Decay']
key_features = [f for f in key_features if f in df.columns]

if len(key_features) >= 3:
    pairplot = sns.pairplot(df[key_features], diag_kind='kde',
                            plot_kws={'alpha': 0.3, 's': 10})
    pairplot.fig.suptitle('Pairplot of Key Features', y=1.02)
    for target in TARGET_FOLDERS:
        pairplot.fig.savefig(image_path(target, "pairplot_key_features.png"), dpi=200)
    print("Saved: pairplot_key_features.png -> compressor/, turbine/")
    plt.close(pairplot.fig)

# =========================================================
# 15. I.I.D. ASSUMPTION CHECK
# =========================================================
print("\n" + "=" * 70)
print("15. I.I.D. ASSUMPTION CHECK")
print("=" * 70)

print("\nThis is a CROSS-SECTIONAL (steady-state) dataset.")
print("Each observation represents an independent snapshot at a particular operating condition.")
print("NO temporal ordering is present - samples are assumed i.i.d.")
print("\nImplications:")
print("  - Time-series methods are NOT applicable")
print("  - Traditional supervised learning (treating samples as independent) is appropriate")
print("  - No autocorrelation concerns")

# Check for any index-based patterns (should be none)
if 'Compressor_Decay' in df.columns:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df['Compressor_Decay'].values[:500], alpha=0.7)
    plt.title('Compressor Decay vs Row Index (first 500)')
    plt.xlabel('Row Index')
    plt.ylabel('Compressor Decay')

    plt.subplot(1, 2, 2)
    plt.plot(df['Turbine_Decay'].values[:500], alpha=0.7, color='orange')
    plt.title('Turbine Decay vs Row Index (first 500)')
    plt.xlabel('Row Index')
    plt.ylabel('Turbine Decay')

    plt.suptitle('Visual Check for Temporal Patterns (should be random)', fontsize=12)
    plt.tight_layout()
    save_for_all_targets("iid_assumption_check.png", dpi=300)
    plt.close()

# =========================================================
# FINAL SUMMARY
# =========================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nCSV tables saved to: {TABLE_DIR}/")
print(f"Figures saved to: {TARGET_FOLDERS['Compressor_Decay']}/")
print(f"Figures saved to: {TARGET_FOLDERS['Turbine_Decay']}/")
print("\nGenerated files:")
print("  - descriptive_statistics.csv")
print("  - high_correlations.csv")
print("  - vif_analysis.csv (if statsmodels installed)")
print("  - decay_by_regime.csv")
print("  - datainfo.png")
print("  - descr_statistics.png")
print("  - target_distributions.png")
print("  - feature_distributions.png")
print("  - feature_boxplots.png")
print("  - correlation_heatmap.png")
print("  - decay_by_speed_regime.png")
print("  - scatter_Compressor_Decay_top_features.png")
print("  - scatter_Turbine_Decay_top_features.png")
print("  - controlled_operating_points.png")
print("  - decay_effects_15knots.png")
print("  - operating_lines.png")
print("  - fuel_consumption_3d_surface.png")
print("  - box_plot_temperatures.png")
print("  - box_plot_pressures.png")
print("  - box_plot_fuel.png")
print("  - pairplot_key_features.png")
print("  - iid_assumption_check.png")

# Clean up temporary column
if 'Speed_Regime' in df.columns:
    df = df.drop(columns=['Speed_Regime'])
