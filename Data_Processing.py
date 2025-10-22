import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Set up non-blocking plots
plt.ion()  # Interactive mode ON

# ----------------------------------------------------------
# Step 1: Download dataset directly from Kaggle
# ----------------------------------------------------------
print("Downloading dataset from Kaggle...")

# Correct way to load dataset with the new KaggleHub API
path = kagglehub.dataset_download("anitarostami/enhanced-gait-biomechanics-dataset")

print(f"Dataset downloaded to: {path}")

# Find the CSV file in the downloaded directory
import os
csv_file = None
for file in os.listdir(path):
    if file.endswith('.csv'):
        csv_file = os.path.join(path, file)
        break

if csv_file is None:
    raise FileNotFoundError("No CSV file found in the downloaded dataset")

print(f"Found CSV file: {csv_file}")

# Load the dataset
df = pd.read_csv(csv_file)

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ----------------------------------------------------------
# Step 2: Filter for knee joint (2) and unbraced condition (1)
# ----------------------------------------------------------
filtered_data = df[(df['joint'] == 2) & (df['condition'] == 1)].copy()

print(f"\nTotal records: {len(df)}")
print(f"Filtered records (knee joint, unbraced): {len(filtered_data)}")
print(f"Left leg samples: {len(filtered_data[filtered_data['leg'] == 1])}")
print(f"Right leg samples: {len(filtered_data[filtered_data['leg'] == 2])}")

# ----------------------------------------------------------
# Step 3: Calculate mean and SD across gait cycle for each leg
# ----------------------------------------------------------
def calculate_leg_averages(data, parameter):
    """Compute mean and SD for each time point and leg."""
    stats = data.groupby(['time', 'leg'])[parameter].agg(['mean', 'std']).reset_index()
    left_leg = stats[stats['leg'] == 1].sort_values('time')
    right_leg = stats[stats['leg'] == 2].sort_values('time')
    return left_leg, right_leg

angle_left, angle_right = calculate_leg_averages(filtered_data, 'angle')
velocity_left, velocity_right = calculate_leg_averages(filtered_data, 'velocity')
accel_left, accel_right = calculate_leg_averages(filtered_data, 'acceleration')

print("\nStatistical calculations completed!")

# ----------------------------------------------------------
# Step 4: Create non-blocking plots
# ----------------------------------------------------------
def plot_leg_data(leg_data, leg_label, parameter_name, y_label, color):
    """Create and display a non-blocking plot"""
    plt.figure(figsize=(10, 6))
    
    time_points = np.linspace(leg_data['time'].min(), leg_data['time'].max(), 200)
    
    mean_spline = make_interp_spline(leg_data['time'], leg_data['mean'], k=3)
    std_spline = make_interp_spline(leg_data['time'], leg_data['std'], k=3)
    
    mean_smooth = mean_spline(time_points)
    std_smooth = std_spline(time_points)
    
    plt.plot(time_points, mean_smooth, color=color, linewidth=2, label=f'{leg_label} mean')
    plt.fill_between(time_points, mean_smooth - std_smooth, mean_smooth + std_smooth,
                    color=color, alpha=0.2, label='±1 SD')
    plt.xlabel('Gait cycle (%)')
    plt.ylabel(y_label)
    plt.title(f'{parameter_name} - {leg_label}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Non-blocking show
    plt.show(block=False)
    plt.pause(0.1)  # Brief pause to render the plot

# Create all plots without blocking
print("\nGenerating plots (non-blocking)...")

# Left leg plots
plot_leg_data(angle_left, 'Left Leg', 'Knee Angle', 'Angle (radians)', 'blue')
plot_leg_data(velocity_left, 'Left Leg', 'Angular Velocity', 'Velocity (rad/s)', 'green')
plot_leg_data(accel_left, 'Left Leg', 'Angular Acceleration', 'Acceleration (rad/s²)', 'purple')

# Right leg plots
plot_leg_data(angle_right, 'Right Leg', 'Knee Angle', 'Angle (radians)', 'red')
plot_leg_data(velocity_right, 'Right Leg', 'Angular Velocity', 'Velocity (rad/s)', 'orange')
plot_leg_data(accel_right, 'Right Leg', 'Angular Acceleration', 'Acceleration (rad/s²)', 'brown')

# ----------------------------------------------------------
# Step 5: Print key statistics with proper positive/negative values
# ----------------------------------------------------------
print("\n" + "="*60)
print("KEY STATISTICS WITH PEAK DIRECTIONS")
print("="*60)

for param_name, left_data, right_data in [
    ('Angle', angle_left, angle_right),
    ('Velocity', velocity_left, velocity_right), 
    ('Acceleration', accel_left, accel_right)
]:
    print(f"\n--- {param_name} ---")
    
    if param_name == 'Angle':
        # For angle, show max flexion and min extension
        left_max_idx = left_data['mean'].idxmax()
        left_min_idx = left_data['mean'].idxmin()
        right_max_idx = right_data['mean'].idxmax()
        right_min_idx = right_data['mean'].idxmin()
        
        print(f"Left leg:")
        print(f"  Max flexion: {left_data.loc[left_max_idx, 'mean']:+.3f} rad at {left_data.loc[left_max_idx, 'time']:.1f}%")
        print(f"  Max extension: {left_data.loc[left_min_idx, 'mean']:+.3f} rad at {left_data.loc[left_min_idx, 'time']:.1f}%")
        
        print(f"Right leg:")
        print(f"  Max flexion: {right_data.loc[right_max_idx, 'mean']:+.3f} rad at {right_data.loc[right_max_idx, 'time']:.1f}%")
        print(f"  Max extension: {right_data.loc[right_min_idx, 'mean']:+.3f} rad at {right_data.loc[right_min_idx, 'time']:.1f}%")
        
    else:
        # For velocity and acceleration, show positive and negative peaks separately
        left_pos_peak_idx = left_data['mean'].idxmax()
        left_neg_peak_idx = left_data['mean'].idxmin()
        right_pos_peak_idx = right_data['mean'].idxmax()
        right_neg_peak_idx = right_data['mean'].idxmin()
        
        left_pos_peak = left_data.loc[left_pos_peak_idx, 'mean']
        left_neg_peak = left_data.loc[left_neg_peak_idx, 'mean']
        right_pos_peak = right_data.loc[right_pos_peak_idx, 'mean']
        right_neg_peak = right_data.loc[right_neg_peak_idx, 'mean']
        
        print(f"Left leg:")
        print(f"  Positive peak: {left_pos_peak:+.3f} at {left_data.loc[left_pos_peak_idx, 'time']:.1f}%")
        print(f"  Negative peak: {left_neg_peak:+.3f} at {left_data.loc[left_neg_peak_idx, 'time']:.1f}%")
        
        print(f"Right leg:")
        print(f"  Positive peak: {right_pos_peak:+.3f} at {right_data.loc[right_pos_peak_idx, 'time']:.1f}%")
        print(f"  Negative peak: {right_neg_peak:+.3f} at {right_data.loc[right_neg_peak_idx, 'time']:.1f}%")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("All plots have been generated and displayed.")
print("The code will continue running without waiting for plots to be closed.")
print("Dataset was downloaded directly from Kaggle.")
print("="*60)

# Optional: Keep the script running until user presses Enter
input("Press Enter to exit and close all plots...")