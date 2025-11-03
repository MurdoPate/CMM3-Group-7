# gait biomechanics — knee joint data (kaggle)
# walking speed ≈ 3.2 ± 0.4 km/h (10 male participants)
# goal: analyse + plot baseline gait → then extrapolate for diff speeds

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os

plt.ion()  # interactive mode on (non-blocking)

# ------------------------------------------------------------
# step 1 → download + load dataset
# ------------------------------------------------------------
print("→ downloading dataset...")

path = kagglehub.dataset_download("anitarostami/enhanced-gait-biomechanics-dataset")
print(f"dataset downloaded to: {path}")

# find csv file
csv = next((os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')), None)
if not csv:
    raise FileNotFoundError("no csv file found")
print(f"found csv: {csv}")

# load to dataframe
df = pd.read_csv(csv)
print(f"loaded ✓  shape {df.shape}")
print(f"columns: {df.columns.tolist()}")

# ------------------------------------------------------------
# step 2 → focus on knee joint + unbraced condition
# ------------------------------------------------------------
data = df[(df['joint'] == 2) & (df['condition'] == 1)].copy()

print(f"\ntotal records: {len(df)}")
print(f"knee (joint=2) + unbraced (cond=1): {len(data)}")
print(f"left leg:  {len(data[data['leg']==1])}")
print(f"right leg: {len(data[data['leg']==2])}")

# ------------------------------------------------------------
# step 3 → mean + std across gait cycle (per leg)
# ------------------------------------------------------------
def leg_stats(d, var):
    g = d.groupby(['time','leg'])[var].agg(['mean','std']).reset_index()
    left = g[g['leg']==1].sort_values('time')
    right = g[g['leg']==2].sort_values('time')
    return left, right

angle_L, angle_R = leg_stats(data, 'angle')
vel_L, vel_R = leg_stats(data, 'velocity')
acc_L, acc_R = leg_stats(data, 'acceleration')

print("\n→ mean ± sd calculated for each leg.")

# ------------------------------------------------------------
# step 4 → make baseline plots (non-blocking)
# ------------------------------------------------------------
def quickplot(d, leg, label, ylabel, col):
    t = np.linspace(d['time'].min(), d['time'].max(), 200)
    mean = make_interp_spline(d['time'], d['mean'], k=3)(t)
    sd = make_interp_spline(d['time'], d['std'], k=3)(t)

    plt.figure(figsize=(10,6))
    plt.plot(t, mean, c=col, lw=2, label=f"{leg} mean")
    plt.fill_between(t, mean - sd, mean + sd, color=col, alpha=0.25, label="±1 SD")
    plt.xlabel("gait cycle (%)")
    plt.ylabel(ylabel)
    plt.title(f"{label} — {leg}")
    plt.legend()
    plt.grid(True, ls='--', alpha=0.7)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

print("\n→ generating baseline plots...")

# left leg plots
quickplot(angle_L, "Left Leg", "Knee Angle", "Angle (rad)", "blue")
quickplot(vel_L, "Left Leg", "Angular Velocity", "Velocity (rad/s)", "green")
quickplot(acc_L, "Left Leg", "Angular Acceleration", "Acceleration (rad/s²)", "purple")

# right leg plots
quickplot(angle_R, "Right Leg", "Knee Angle", "Angle (rad)", "red")
quickplot(vel_R, "Right Leg", "Angular Velocity", "Velocity (rad/s)", "orange")
quickplot(acc_R, "Right Leg", "Angular Acceleration", "Acceleration (rad/s²)", "brown")

# ------------------------------------------------------------
# step 5 → print peak stats (flexion / extension)
# ------------------------------------------------------------
print("\n" + "="*60)
print("KNEE JOINT PEAK ANGLES")
print("="*60)

def peaks(name, L, R):
    lmax, lmin = L['mean'].idxmax(), L['mean'].idxmin()
    rmax, rmin = R['mean'].idxmax(), R['mean'].idxmin()
    print(f"\n{name}:")
    print(f"→ Left:  flex {L.loc[lmax,'mean']:+.3f} rad @ {L.loc[lmax,'time']:.1f}% | ext {L.loc[lmin,'mean']:+.3f} rad @ {L.loc[lmin,'time']:.1f}%")
    print(f"→ Right: flex {R.loc[rmax,'mean']:+.3f} rad @ {R.loc[rmax,'time']:.1f}% | ext {R.loc[rmin,'mean']:+.3f} rad @ {R.loc[rmin,'time']:.1f}%")

peaks("Knee Angle", angle_L, angle_R)

print("\n→ baseline analysis done ✓")

# ------------------------------------------------------------
# step 6 → extrapolate to diff walking speeds
# ------------------------------------------------------------
def gait_scale(d, base_v, new_v):
    r = new_v / base_v
    t_scaled = d['time'] / r**0.8  # faster → shorter cycle
    t_scaled = (t_scaled / t_scaled.max()) * 100  # normalize to 0–100%
    amp = 1 + 0.15 * (r - 1)  # amplitude grows slightly w/ speed
    return pd.DataFrame({
        'time': t_scaled,
        'mean': d['mean'] * amp,
        'std': d['std'] * amp
    })

speeds = [2.5, 3.2, 3.8, 4.5]
print("\n→ extrapolating knee angles for diff speeds...")

# ------------------------------------------------------------
# step 7 → plot knee angle vs gait cycle (left + right)
# ------------------------------------------------------------
# left
plt.figure(figsize=(10,6))
for v in speeds:
    scaled = gait_scale(angle_L, 3.2, v)
    plt.plot(scaled['time'], scaled['mean'], lw=2, label=f"{v:.1f} km/h")
plt.title("Left Knee Angle vs Gait Cycle — Different Walking Speeds")
plt.xlabel("Gait cycle (%)")
plt.ylabel("Knee angle (rad)")
plt.grid(True, ls='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# right
plt.figure(figsize=(10,6))
for v in speeds:
    scaled = gait_scale(angle_R, 3.2, v)
    plt.plot(scaled['time'], scaled['mean'], lw=2, label=f"{v:.1f} km/h")
plt.title("Right Knee Angle vs Gait Cycle — Different Walking Speeds")
plt.xlabel("Gait cycle (%)")
plt.ylabel("Knee angle (rad)")
plt.grid(True, ls='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

print("\n✅ done — all 6 baseline + 2 extrapolated plots generated.")
input("press Enter to close plots...")
