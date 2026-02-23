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

# Load original data and reverse row order to correct time-index
# (Dataset was discovered to be reversed with respect to time)
df_original = pd.read_csv(BASE_DIR / 'cleaned_data.csv')
df = df_original.iloc[::-1].reset_index(drop=True)

# Save time-corrected dataset
df.to_csv(BASE_DIR / 'Time-index-data.csv', index=False)
print("Created: Time-index-data.csv (reversed row order for correct temporal sequence)")


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

print("\n--- Lifecycle Structure ---")
n_speeds = df['Ship_Speed'].nunique() if 'Ship_Speed' in df.columns else 9
n_compressor_levels = df['Compressor_Decay'].nunique() if 'Compressor_Decay' in df.columns else 51
n_turbine_levels = df['Turbine_Decay'].nunique() if 'Turbine_Decay' in df.columns else 26
turbine_cycle_length = n_speeds * n_turbine_levels
compressor_lifecycle = n_speeds * n_turbine_levels * n_compressor_levels
print(f"  Ship Speed levels: {n_speeds}")
print(f"  Compressor Decay levels: {n_compressor_levels} (0.95 → 1.0)")
print(f"  Turbine Decay levels: {n_turbine_levels} (0.975 → 1.0)")
print(f"  Turbine maintenance cycle: ~{turbine_cycle_length} units (rows)")
print(f"  Compressor full lifecycle: ~{compressor_lifecycle} units (rows)")
print(f"  Total turbine cycles per compressor lifecycle: ~{n_compressor_levels}")

# Data info summary figure used in appendix
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')
info_text = (
    f"Rows: {df.shape[0]}\n"
    f"Columns: {df.shape[1]}\n"
    f"Numeric columns: {df.select_dtypes(include=[np.number]).shape[1]}\n"
    f"Missing values: {int(df.isna().sum().sum())}\n"
    f"Duplicate rows: {int(df.duplicated().sum())}\n"
    f"Constant columns: {', '.join([c for c in df.columns if df[c].nunique() <= 1])}\n\n"
    f"Lifecycle Structure:\n"
    f"  Turbine cycle: ~{turbine_cycle_length} units (234 rows per maintenance)\n"
    f"  Compressor lifecycle: ~{compressor_lifecycle} units (~11,934 rows total)"
)
ax.text(0.02, 0.95, "Dataset Information", fontsize=16, fontweight='bold', va='top')
ax.text(0.02, 0.80, info_text, fontsize=11, va='top')
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
# 15. DATA STRUCTURE ANALYSIS (FACTORIAL DESIGN)
# =========================================================
print("\n" + "=" * 70)
print("15. DATA STRUCTURE ANALYSIS (FACTORIAL DESIGN)")
print("=" * 70)

print("\nThis dataset follows a FACTORIAL DESIGN with temporal interpretation:")
print("  - 9 ship speeds × 26 turbine decay levels × 51 compressor decay levels = 11,934 rows")
print("\nKey Lifecycle Information:")
print("  - COMPRESSOR: Single lifecycle of ~11,934 units (rows) before maintenance required")
print("    * Decay range: 1.0 (new) → 0.95 (threshold) over 51 decay levels")
print("    * Each decay level appears 234 times (9 speeds × 26 turbine states)")
print("  - TURBINE: ~51 maintenance cycles of ~234 units each")
print("    * Decay range: 1.0 (new) → 0.975 (threshold) over 26 decay levels per cycle")
print("    * Each cycle = 9 speeds × 26 decay levels = 234 rows")
print("\nImplications:")
print("  - Data reversed to correct temporal ordering (index 0 = new, higher = worn)")
print("  - Turbine cycles ~51× faster than compressor")
print("  - RUL predictions are in units (rows) matching the time index")

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

    plt.suptitle('Factorial Design Temporal Structure (showing degradation patterns)', fontsize=12)
    plt.tight_layout()
    save_for_all_targets("iid_assumption_check.png", dpi=300)
    plt.close()

# =========================================================
# 16. DECAY VS INDEX GRAPHS
# =========================================================
print("\n" + "=" * 70)
print("16. DECAY VS INDEX GRAPHS")
print("=" * 70)
print("\nVisualization of degradation over time index:")
print("  - Compressor: Single continuous degradation over ~11,934 units")
print("  - Turbine: Sawtooth pattern showing ~51 maintenance cycles of ~234 units each")

# Graph 1: Compressor Decay vs Index
if 'Compressor_Decay' in df.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Compressor_Decay'], alpha=0.7, color='blue', linewidth=0.5)
    plt.scatter(df.index, df['Compressor_Decay'], alpha=0.3, s=5, color='blue')
    plt.title('Compressor Decay vs Index Number', fontsize=14)
    plt.xlabel('Index Number', fontsize=12)
    plt.ylabel('Compressor Decay', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_for_all_targets("compressor_decay_vs_index.png", dpi=300)
    plt.close()
    print("Saved: compressor_decay_vs_index.png")

# Graph 2: Turbine Decay vs Index
if 'Turbine_Decay' in df.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Turbine_Decay'], alpha=0.7, color='red', linewidth=0.5)
    plt.scatter(df.index, df['Turbine_Decay'], alpha=0.3, s=5, color='red')
    plt.title('Turbine Decay vs Index Number', fontsize=14)
    plt.xlabel('Index Number', fontsize=12)
    plt.ylabel('Turbine Decay', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_for_all_targets("turbine_decay_vs_index.png", dpi=300)
    plt.close()
    print("Saved: turbine_decay_vs_index.png")

# =========================================================
# 17. REMAINING USEFUL LIFE (RUL) PREDICTION
# =========================================================
print("\n" + "=" * 70)
print("17. REMAINING USEFUL LIFE (RUL) PREDICTION")
print("=" * 70)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Define thresholds
COMPRESSOR_THRESHOLD = 0.95
TURBINE_THRESHOLD = 0.975

# Select snapshot from line 997
snapshot_idx = 997
snapshot = df.loc[[snapshot_idx]]
snapshot_compressor = snapshot['Compressor_Decay'].values[0]
snapshot_turbine = snapshot['Turbine_Decay'].values[0]

print(f"\nSnapshot Selected (Index {snapshot_idx}):")
print(f"  Compressor Decay: {snapshot_compressor:.4f}")
print(f"  Turbine Decay: {snapshot_turbine:.4f}")

# --- TRAIN RANDOM FOREST MODELS (same as models.py) ---
print("\nTraining Random Forest models for RUL prediction...")

# Prepare features (same as models.py) - exclude non-numeric and target columns
exclude_cols = ['Compressor_Decay', 'Turbine_Decay', 'T1', 'P1', 'Speed_Regime']
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
X_all = df[feature_cols]
scaler = StandardScaler()

# Train Compressor model
y_comp_all = df['Compressor_Decay']
X_scaled = scaler.fit_transform(X_all)
rf_comp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_comp.fit(X_scaled, y_comp_all)
print(f"  Compressor RF trained (R² on full data: {rf_comp.score(X_scaled, y_comp_all):.4f})")

# Train Turbine model
y_turb_all = df['Turbine_Decay']
rf_turb = RandomForestRegressor(n_estimators=100, random_state=42)
rf_turb.fit(X_scaled, y_turb_all)
print(f"  Turbine RF trained (R² on full data: {rf_turb.score(X_scaled, y_turb_all):.4f})")

# --- COMPRESSOR RUL PREDICTION USING RF MODEL ---
historical_data = df[df.index <= snapshot_idx].copy()

# Get degradation rate from linear regression on historical data
X_time_comp = historical_data.index.values.reshape(-1, 1)
y_decay_comp = historical_data['Compressor_Decay'].values
lr_comp = LinearRegression()
lr_comp.fit(X_time_comp, y_decay_comp)

# Use RF model to predict current decay
snapshot_features = scaler.transform(snapshot[feature_cols])
rf_predicted_comp = rf_comp.predict(snapshot_features)[0]

# Extrapolate using degradation rate to find failure time
if lr_comp.coef_[0] != 0:
    comp_failure_time = (COMPRESSOR_THRESHOLD - lr_comp.intercept_) / lr_comp.coef_[0]
    comp_current_time = snapshot_idx
    compressor_rul = max(0, comp_failure_time - comp_current_time)
else:
    compressor_rul = float('inf')
    comp_failure_time = len(df) * 1.5

print(f"\nCompressor RUL Prediction (Random Forest + Linear Extrapolation):")
print(f"  Actual decay: {snapshot_compressor:.4f}")
print(f"  RF predicted decay: {rf_predicted_comp:.4f}")
print(f"  Threshold: {COMPRESSOR_THRESHOLD}")
print(f"  Degradation rate: {lr_comp.coef_[0]:.6f} per time unit")
print(f"  Predicted RUL: {compressor_rul:.1f} time units")

# --- TURBINE RUL PREDICTION USING RF MODEL ---
# Find current maintenance cycle
historical_turbine = df[df.index <= snapshot_idx].copy()
turbine_decay = historical_turbine['Turbine_Decay'].values
maintenance_indices = [0]
for i in range(1, len(turbine_decay)):
    if turbine_decay[i] == 1.0 and turbine_decay[i-1] < 1.0:
        maintenance_indices.append(historical_turbine.index[i])

cycle_start = maintenance_indices[-1]
for i, maint_idx in enumerate(maintenance_indices):
    if maint_idx > snapshot_idx:
        cycle_start = maintenance_indices[i-1]
        break

current_cycle = historical_turbine[historical_turbine.index >= cycle_start].copy()
print(f"\nTurbine cycle detected: starts at index {cycle_start}, {len(current_cycle)} points in cycle")

# Fit linear regression on current cycle
X_time_turb = current_cycle.index.values.reshape(-1, 1)
y_decay_turb = current_cycle['Turbine_Decay'].values
lr_turb = LinearRegression()
lr_turb.fit(X_time_turb, y_decay_turb)

# Use RF model to predict current decay
rf_predicted_turb = rf_turb.predict(snapshot_features)[0]

if lr_turb.coef_[0] != 0:
    turb_failure_time = (TURBINE_THRESHOLD - lr_turb.intercept_) / lr_turb.coef_[0]
    turb_current_time = snapshot_idx
    turbine_rul = max(0, turb_failure_time - turb_current_time)
else:
    turbine_rul = float('inf')
    turb_failure_time = cycle_start + 500

print(f"\nTurbine RUL Prediction (Random Forest + Linear Extrapolation):")
print(f"  Actual decay: {snapshot_turbine:.4f}")
print(f"  RF predicted decay: {rf_predicted_turb:.4f}")
print(f"  Threshold: {TURBINE_THRESHOLD}")
print(f"  Degradation rate: {lr_turb.coef_[0]:.6f} per row")
print(f"  Predicted RUL: {turbine_rul:.0f} units (rows)")

# --- LINEAR REGRESSION PLOT (separate) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subsample data for visualization (every 100th point)
sample_step = max(1, len(historical_data) // 100)
X_comp_plot = X_time_comp[::sample_step]
y_comp_plot = y_decay_comp[::sample_step]

# Compressor Linear Regression - show full lifecycle extrapolation
ax1 = axes[0]
ax1.scatter(X_comp_plot, y_comp_plot, alpha=0.7, s=50, color='blue', label='Historical Data Points')
# Extend x-axis to show full extrapolation to failure
x_fit = np.linspace(0, comp_failure_time * 1.05, 100).reshape(-1, 1)
y_fit = lr_comp.predict(x_fit)
ax1.plot(x_fit, y_fit, 'b-', linewidth=2, label=f'Linear Fit (slope={lr_comp.coef_[0]:.6f})')
# Mark RF prediction at snapshot
ax1.scatter([snapshot_idx], [rf_predicted_comp], color='orange', s=200, zorder=5,
            marker='*', edgecolors='black', linewidth=1, label=f'RF Prediction ({rf_predicted_comp:.4f})')
# Mark threshold
ax1.axhline(y=COMPRESSOR_THRESHOLD, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Threshold ({COMPRESSOR_THRESHOLD})')
ax1.set_xlabel('Time Index (units)', fontsize=12)
ax1.set_ylabel('Compressor Decay', fontsize=12)
ax1.set_title('Compressor Decay - Linear Extrapolation (~11,934 unit lifecycle)', fontsize=14)
ax1.set_xlim(0, comp_failure_time * 1.1)
ax1.set_ylim(0.93, 1.02)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)

# Turbine Linear Regression - show in relative cycle position (0 to ~234)
ax2 = axes[1]
# Convert to relative positions within cycle
X_turb_relative = X_time_turb - cycle_start
sample_step_t = max(1, len(current_cycle) // 100)
X_turb_plot = X_turb_relative[::sample_step_t]
y_turb_plot = y_decay_turb[::sample_step_t]

ax2.scatter(X_turb_plot, y_turb_plot, alpha=0.7, s=50, color='red', label='Current Cycle Data')
# Extrapolation line in relative position
failure_pos_relative = turb_failure_time - cycle_start
x_fit_t = np.linspace(0, failure_pos_relative * 1.1, 100).reshape(-1, 1)
# Predict using absolute indices, then plot with relative x
x_fit_t_abs = x_fit_t + cycle_start
y_fit_t = lr_turb.predict(x_fit_t_abs)
ax2.plot(x_fit_t, y_fit_t, 'r-', linewidth=2, label=f'Linear Fit (slope={lr_turb.coef_[0]:.6f})')
# Mark RF prediction at snapshot (relative position)
snapshot_relative = snapshot_idx - cycle_start
ax2.scatter([snapshot_relative], [rf_predicted_turb], color='orange', s=200, zorder=5,
            marker='*', edgecolors='black', linewidth=1, label=f'RF Prediction ({rf_predicted_turb:.4f})')
# Mark threshold
ax2.axhline(y=TURBINE_THRESHOLD, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Threshold ({TURBINE_THRESHOLD})')
ax2.set_xlabel('Cycle Position (units)', fontsize=12)
ax2.set_ylabel('Turbine Decay', fontsize=12)
ax2.set_title('Turbine Decay - Linear Extrapolation (~234 unit cycle)', fontsize=14)
ax2.set_xlim(0, 300)
ax2.set_ylim(0.93, 1.02)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
save_for_all_targets("linear_regression_fit.png", dpi=300)
plt.close()
print("\nSaved: linear_regression_fit.png")

# --- RUL PREDICTION PLOT (one lifecycle) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Compressor RUL Plot - full lifecycle in time index
ax1 = axes[0]
x_lifecycle = np.linspace(0, comp_failure_time * 1.05, 100).reshape(-1, 1)
y_lifecycle = lr_comp.predict(x_lifecycle)
ax1.plot(x_lifecycle, y_lifecycle, 'b-', linewidth=2, label='Degradation Path')
# Plot current state point
ax1.scatter([snapshot_idx], [snapshot_compressor], color='green', s=150, zorder=5,
            edgecolors='black', linewidth=2, label=f'Current State ({snapshot_compressor:.3f})')
# Plot threshold line
ax1.axhline(y=COMPRESSOR_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold ({COMPRESSOR_THRESHOLD})')
# Mark failure point
ax1.scatter([comp_failure_time], [COMPRESSOR_THRESHOLD], color='red', s=150, zorder=5,
            marker='X', edgecolors='black', linewidth=2, label='Predicted Failure')
# Annotate RUL
ax1.annotate(f'RUL ≈ {compressor_rul:.0f} units',
             xy=(comp_failure_time, COMPRESSOR_THRESHOLD),
             xytext=(snapshot_idx + compressor_rul * 0.3, COMPRESSOR_THRESHOLD + 0.012),
             fontsize=11, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red'))

ax1.set_xlabel('Time Index (CSV Row)', fontsize=12)
ax1.set_ylabel('Compressor Decay', fontsize=12)
ax1.set_title('Compressor Remaining Useful Life (RUL) Prediction', fontsize=14)
ax1.set_xlim(0, comp_failure_time * 1.1)
ax1.set_ylim(0.93, 1.02)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)

# Turbine RUL Plot - single cycle (relative to cycle start, 0 to 300)
ax2 = axes[1]
# Convert to relative position within cycle (0 = cycle start)
current_pos_in_cycle = snapshot_idx - cycle_start
failure_pos_in_cycle = turb_failure_time - cycle_start

x_lifecycle_t = np.linspace(0, failure_pos_in_cycle * 1.05, 100).reshape(-1, 1)
# Need to predict using absolute indices, then plot with relative x
x_lifecycle_t_abs = x_lifecycle_t + cycle_start
y_lifecycle_t = lr_turb.predict(x_lifecycle_t_abs)
ax2.plot(x_lifecycle_t, y_lifecycle_t, 'r-', linewidth=2, label='Degradation Path')
# Plot current state point (relative position)
ax2.scatter([current_pos_in_cycle], [snapshot_turbine], color='green', s=150, zorder=5,
            edgecolors='black', linewidth=2, label=f'Current State ({snapshot_turbine:.3f})')
ax2.axhline(y=TURBINE_THRESHOLD, color='darkred', linestyle='--', linewidth=2, label=f'Threshold ({TURBINE_THRESHOLD})')
ax2.scatter([failure_pos_in_cycle], [TURBINE_THRESHOLD], color='darkred', s=150, zorder=5,
            marker='X', edgecolors='black', linewidth=2, label='Predicted Failure')
ax2.annotate(f'RUL ≈ {turbine_rul:.0f} units',
             xy=(failure_pos_in_cycle, TURBINE_THRESHOLD),
             xytext=(current_pos_in_cycle + 20, TURBINE_THRESHOLD + 0.008),
             fontsize=11, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='darkred'))

ax2.set_xlabel('Cycle Position (units)', fontsize=12)
ax2.set_ylabel('Turbine Decay', fontsize=12)
ax2.set_title('Turbine Remaining Useful Life (RUL) Prediction (Single Cycle)', fontsize=14)
ax2.set_xlim(0, 300)
ax2.set_ylim(0.93, 1.02)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
save_for_all_targets("rul_prediction.png", dpi=300)
plt.close()
print("Saved: rul_prediction.png")

# Save RUL metrics to CSV
rul_metrics = pd.DataFrame({
    'Component': ['Compressor', 'Turbine'],
    'Snapshot_Index': [snapshot_idx, snapshot_idx],
    'Current_Decay': [snapshot_compressor, snapshot_turbine],
    'Threshold': [COMPRESSOR_THRESHOLD, TURBINE_THRESHOLD],
    'Predicted_RUL': [compressor_rul, turbine_rul],
    'Regression_Slope': [lr_comp.coef_[0], lr_turb.coef_[0]],
    'Regression_Intercept': [lr_comp.intercept_, lr_turb.intercept_]
})
rul_metrics.to_csv(table_path("lifecycle_analysis.csv"), index=False)
print("Saved: lifecycle_analysis.csv")

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
print("  - lifecycle_analysis.csv")
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
print("  - compressor_decay_vs_index.png")
print("  - turbine_decay_vs_index.png")
print("  - linear_regression_fit.png")
print("  - rul_prediction.png")

# Clean up temporary column
if 'Speed_Regime' in df.columns:
    df = df.drop(columns=['Speed_Regime'])
