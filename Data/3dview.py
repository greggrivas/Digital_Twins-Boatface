import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from scipy.interpolate import RegularGridInterpolator
import os

# Load your data (use path relative to script location)
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, 'cleaned_data.csv')) 

# 1. Select a high-speed operating point (e.g., 27 knots)
# This is where the engine is under high load and health differences are most visible
speed_val = 15
df_3d = df[df['Ship_Speed'] == speed_val]

# 2. Prepare the Grid (Pivot the data)
# X = Turbine Decay, Y = Compressor Decay, Z = Fuel Flow
pivot = df_3d.pivot_table(values='Fuel_Flow', index='Compressor_Decay', columns='Turbine_Decay')

X_vals = pivot.columns.values
Y_vals = pivot.index.values
X, Y = np.meshgrid(X_vals, Y_vals)
Z = pivot.values

# Create the Color Grid (Turbine Temperature T48)
# This makes the plot "4D": X, Y, Z (height) + Color
pivot_temp = df_3d.pivot_table(values='T48', index='Compressor_Decay', columns='Turbine_Decay')
Temp = pivot_temp.values

# Normalize Temperature values to a 0-1 range for the colormap
min_temp, max_temp = Temp.min(), Temp.max()
Temp_norm = (Temp - min_temp) / (max_temp - min_temp)

# 3. Create the Advanced 3D Plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot a smooth surface with color representing Turbine Temperature (T48)
# Blue = Lower Temp | Red = Higher Temp (using coolwarm colormap)
surf = ax.plot_surface(X, Y, Z, facecolors=cm.coolwarm(Temp_norm), alpha=0.8, edgecolor='none', antialiased=True, shade=False)

# Add a subtle wireframe to give it a "digital grid" look
ax.plot_wireframe(X, Y, Z, color='black', alpha=0.1, linewidth=0.5)

# 4. AXIS LOGIC: We invert the axes so '1.0' (New) is at the back 
# and '0.95' (Worn) is at the front. 
# This makes the "climb" in fuel consumption look like a mountain we are rising up.
ax.set_xlim(1.0, 0.975)
ax.set_ylim(1.0, 0.95)

# Labels with LaTeX for a professional look
ax.set_xlabel('Turbine Decay ($k_{mt}$)', fontsize=12, labelpad=10)
ax.set_ylabel('Compressor Decay ($k_{mc}$)', fontsize=12, labelpad=10)
ax.set_zlabel('Fuel Flow (kg/s)', fontsize=12, labelpad=10)
ax.set_title(f'Digital Twin: Fuel Consumption Surface at {speed_val} Knots\n(Height = Fuel Flow, Color = Turbine Temperature)', fontsize=15, pad=20)

# Add a color bar for Turbine Temperature
sm = cm.ScalarMappable(cmap=cm.coolwarm)
sm.set_array(Temp)
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Turbine Exit Temperature T48 (°C)', rotation=270, labelpad=20)

# Set the perspective (Elev = height angle, Azim = rotation)
ax.view_init(elev=30, azim=135)

plt.tight_layout()

# =============================================================================
# SAVE CLEAN FIGURE FOR REPORT (without marker)
# =============================================================================
plt.savefig(os.path.join(script_dir, "advanced_engine_surface_3d.png"), dpi=300)
print(f"Report figure saved: {os.path.join(script_dir, 'advanced_engine_surface_3d.png')}")

# =============================================================================
# 5. CURRENT ENGINE STATE MARKER (Using trained ML models)
# =============================================================================
from sklearn.ensemble import RandomForestRegressor

# Prepare features and targets for model training
feature_cols = [col for col in df.columns if col not in ['Compressor_Decay', 'Turbine_Decay', 'T1', 'P1']]
X_all = df[feature_cols]
y_compressor = df['Compressor_Decay']
y_turbine = df['Turbine_Decay']

# Train Random Forest models (same as in models.py)
print("Training Random Forest models...")
rf_compressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_turbine = RandomForestRegressor(n_estimators=100, random_state=42)
rf_compressor.fit(X_all, y_compressor)
rf_turbine.fit(X_all, y_turbine)
print("Models trained!")

# =============================================================================
# SIMULATE "LIVE" SENSOR DATA
# =============================================================================
# In a real HMI, this would come from the vessel's data acquisition system.
# Here we pick a random sample from the dataset to simulate live readings.
np.random.seed(None)  # Use current time for randomness
sample_idx = np.random.randint(0, len(df))
live_sensor_data = X_all.iloc[[sample_idx]]

# Get the actual decay values for comparison (in real life you wouldn't have these)
actual_compressor = y_compressor.iloc[sample_idx]
actual_turbine = y_turbine.iloc[sample_idx]

# Predict decay coefficients using the trained models
current_compressor_decay = rf_compressor.predict(live_sensor_data)[0]
current_turbine_decay = rf_turbine.predict(live_sensor_data)[0]

print(f"\n--- LIVE ENGINE STATE (Sample #{sample_idx}) ---")
print(f"Ship Speed: {live_sensor_data['Ship_Speed'].values[0]:.1f} knots")
print(f"Predicted Compressor Decay: {current_compressor_decay:.4f} (Actual: {actual_compressor:.4f})")
print(f"Predicted Turbine Decay:    {current_turbine_decay:.4f} (Actual: {actual_turbine:.4f})")

# Interpolate the fuel flow and temperature for the current state from the surface
# Create interpolator (note: Y_vals = compressor, X_vals = turbine)
interp_fuel = RegularGridInterpolator((Y_vals, X_vals), Z, method='linear', bounds_error=False, fill_value=None)
interp_temp = RegularGridInterpolator((Y_vals, X_vals), Temp, method='linear', bounds_error=False, fill_value=None)

current_fuel = interp_fuel((current_compressor_decay, current_turbine_decay))
current_temp = interp_temp((current_compressor_decay, current_turbine_decay))

print(f"Interpolated Fuel Flow: {current_fuel:.3f} kg/s")
print(f"Interpolated T48: {current_temp:.1f} °C")

# Plot the current state as a prominent 3D marker
ax.scatter(
    [current_turbine_decay],
    [current_compressor_decay],
    [current_fuel],
    color='lime',
    s=200,           # Size of marker
    edgecolors='black',
    linewidths=2,
    marker='o',
    zorder=10,
    label=f'Current State\nTurbine: {current_turbine_decay:.3f}\nCompressor: {current_compressor_decay:.3f}\nFuel: {current_fuel:.3f} kg/s\nT48: {current_temp:.1f} °C'
)

# Add a vertical drop line from the point to the base for depth perception
ax.plot(
    [current_turbine_decay, current_turbine_decay],
    [current_compressor_decay, current_compressor_decay],
    [Z.min(), current_fuel],
    color='lime', linestyle='--', linewidth=1.5, alpha=0.7
)

# Add legend
ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.show()