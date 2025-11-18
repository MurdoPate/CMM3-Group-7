# ============================
# Full pipeline: data + model (Fixed to 3.2 km/h experimental data)
# ============================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import make_interp_spline
import kagglehub

plt.ion()

# ----------------------------------------
# 1) Load gait dataset (Kaggle) - 3.2 km/h data
# ----------------------------------------
print("→ Downloading gait dataset (3.2 km/h walking data)...")
path = kagglehub.dataset_download("anitarostami/enhanced-gait-biomechanics-dataset")
csv = next((os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")), None)
if not csv:
    raise FileNotFoundError("No CSV found in dataset path.")

df = pd.read_csv(csv)
data = df[(df["joint"] == 2) & (df["condition"] == 1)].copy()

def leg_stats(d, var):
    g = d.groupby(["time", "leg"])[var].agg(["mean", "std"]).reset_index()
    left  = g[g["leg"] == 1].sort_values("time")
    right = g[g["leg"] == 2].sort_values("time")
    return left, right

angle_L, _ = leg_stats(data, "angle")
print("✓ Dataset loaded and filtered (knee, unbraced, 3.2 km/h)")

# ----------------------------------------
# 2) Reference gait from dataset (phase 0..1) - 3.2 km/h pattern
# ----------------------------------------
phi_ref_raw   = angle_L["time"].to_numpy(dtype=float)
theta_ref_raw = angle_L["mean"].to_numpy(dtype=float)

phi_ref, keep_idx = np.unique(phi_ref_raw, return_index=True)
theta_ref_exp = theta_ref_raw[keep_idx]

def theta_ref(phi):
    return np.interp(np.mod(phi, 1.0), phi_ref, theta_ref_exp)

dtheta_dphi_samples = np.gradient(theta_ref_exp, phi_ref)
def dtheta_dphi(phi):
    return np.interp(np.mod(phi, 1.0), phi_ref, dtheta_dphi_samples)

exp_time_pct  = phi_ref * 100.0
exp_angle_deg = np.rad2deg(theta_ref_exp)
theta0_mean = float(np.mean(theta_ref(phi_ref)))

# ----------------------------------------
# 3) Geometry from height + PROSTHETIC inertia
# ----------------------------------------
try:
    height = float(input("Enter your height in cm: ").strip() or "178")
except ValueError:
    height = 178.0
print(f"Using height = {height:.1f} cm")

# ----------------------------------------
# 4) Walking Speed - FIXED to match experimental data
# ----------------------------------------
speed_kmh = 3.2  # Fixed to match the experimental dataset
walking_speed = speed_kmh / 3.6
print(f"Using experimental walking speed: {speed_kmh} km/h")

# Prosthetic mass and constants
m = 4.0  # Mass of PROSTHETIC device (kg), not user body mass
g = 9.81

lower_leg_length = (height - 123) / 1.24
foot_size        = (0.044 * height) + 19.286
stride_length_cm = 0.415 * height
stride_length_m  = stride_length_cm / 100.0

L1 = lower_leg_length / 100.0
L2 = foot_size / 100.0

den   = 2 * (lower_leg_length + foot_size)
x_bar = (foot_size**2) / den
y_bar = (lower_leg_length**2) / den
COM_to_A = lower_leg_length - y_bar
L3 = COM_to_A / 100.0

I_A = (
    m * (1.0 / (lower_leg_length + foot_size)) *
    ((1/3)*lower_leg_length**3 + lower_leg_length**2*foot_size + (1/3)*foot_size**3)
) / 1e4

print("\n--- Geometry ---")
print(f"L1={lower_leg_length:.2f} cm  L2={foot_size:.2f} cm  L3={COM_to_A:.2f} cm ({L3:.3f} m)")
print(f"I_A={I_A:.6f} kg·m²")

# ----------------------------------------
# 5) Target speed and stiffness - Based on 3.2 km/h
# ----------------------------------------
f_target = walking_speed / stride_length_m
T_cycle  = 1.0 / f_target
w_target = 2*np.pi*f_target

print(f"\n--- Target gait ---")
print(f"Stride={stride_length_m:.3f} m  v={walking_speed:.3f} m/s  f={f_target:.3f} Hz  T_cycle={T_cycle:.3f} s")

k_star = (I_A * w_target**2) - (m * g * L3)
print(f"\nCalculated stiffness from height & cadence (k*): {k_star:.3f} N·m/rad")

# Soft passive stiffness
k_soft = 5.0
print(f"Using soft stiffness for prosthetic joint: k_soft = {k_soft:.2f} N·m/rad")

# ----------------------------------------
# 6) Dynamics + simulators
# ----------------------------------------
def knee_rhs(t, y, p):
    theta, theta_dot = y
    k, c, f, theta0, Kp, Kd, m_, g_, L3_, I_ = (
        p["k"], p["c"], p["f"], p["theta0"], p["Kp"], p["Kd"], p["m"], p["g"], p["L3"], p["I"]
    )
    tau_joint = -k*(theta - theta0) - c*theta_dot
    tau_g     = m_ * g_ * L3_ * np.sin(theta)
    phi = (f*t) % 1.0
    th_des     = theta_ref(phi)
    th_des_dot = dtheta_dphi(phi) * f
    tau_hip = Kp*(th_des - theta) + Kd*(th_des_dot - theta_dot)
    theta_ddot = (tau_hip - tau_g + tau_joint) / I_
    return [theta_dot, theta_ddot]

def simulate_cycles(k, c, f, Kp=200.0, Kd=15.0, theta0=theta0_mean, ncycles=8, n_per_cycle=400):
    """Simulate multiple gait cycles with consistent sampling per cycle"""
    T = 1.0 / f
    total_time = ncycles * T
    p = dict(k=k, c=c, f=f, Kp=Kp, Kd=Kd, theta0=theta0, m=m, g=g, L3=L3, I=I_A)
    y0 = [theta_ref(0.0), dtheta_dphi(0.0) * f]
    
    # High-resolution simulation
    sol = solve_ivp(lambda t, y: knee_rhs(t, y, p),
                    [0.0, total_time], y0, method="RK45",
                    rtol=1e-7, atol=1e-9, max_step=T/1000)
    
    # Sample each cycle consistently
    t_samples = []
    theta_samples = []
    thetad_samples = []
    theta_ref_samples = []
    
    for cycle in range(ncycles):
        t_cycle = np.linspace(cycle*T, (cycle+1)*T, n_per_cycle, endpoint=False)
        theta_cycle = np.interp(t_cycle, sol.t, sol.y[0])
        thetad_cycle = np.interp(t_cycle, sol.t, sol.y[1])
        phi_cycle = (f * t_cycle) % 1.0
        theta_ref_cycle = theta_ref(phi_cycle)
        
        t_samples.append(t_cycle)
        theta_samples.append(theta_cycle)
        thetad_samples.append(thetad_cycle)
        theta_ref_samples.append(theta_ref_cycle)
    
    return (np.array(t_samples), np.array(theta_samples), 
            np.array(thetad_samples), np.array(theta_ref_samples))

def energy_damped(theta_dot, c, t):
    return float(c * trapezoid(theta_dot**2, t))

def analyze_multiple_cycles(theta_cycles, theta_ref_cycles, thetad_cycles, t_cycles, c):
    """Analyze statistics across multiple gait cycles"""
    n_cycles = len(theta_cycles)
    
    # Individual cycle metrics
    rms_per_cycle = []
    energy_per_cycle = []
    
    for i in range(n_cycles):
        # RMS error for this cycle
        rms_deg = np.rad2deg(np.sqrt(np.mean((theta_cycles[i] - theta_ref_cycles[i])**2)))
        rms_per_cycle.append(rms_deg)
        
        # Energy dissipated this cycle
        energy = energy_damped(thetad_cycles[i], c, t_cycles[i] - t_cycles[i][0])
        energy_per_cycle.append(energy)
    
    # Statistics across cycles
    rms_mean = np.mean(rms_per_cycle)
    rms_std = np.std(rms_per_cycle)
    energy_mean = np.mean(energy_per_cycle)
    energy_std = np.std(energy_per_cycle)
    
    # Convergence check (compare last 3 cycles)
    if n_cycles >= 3:
        last_3_rms = rms_per_cycle[-3:]
        rms_convergence = np.std(last_3_rms) / np.mean(last_3_rms)
    else:
        rms_convergence = float('inf')
    
    return {
        'rms_per_cycle': rms_per_cycle,
        'energy_per_cycle': energy_per_cycle,
        'rms_mean': rms_mean,
        'rms_std': rms_std,
        'energy_mean': energy_mean,
        'energy_std': energy_std,
        'convergence_ratio': rms_convergence
    }

# Smooth reference curve
t_s = np.linspace(0, 100, 300)
ang_s = make_interp_spline(exp_time_pct, exp_angle_deg, k=3)(t_s)

# ----------------------------------------
# 7) Coarse→Fine δ sweep with multiple cycle analysis
# ----------------------------------------
Kp, Kd = 200.0, 15.0

def rms_for_delta(dlt, ncycles=6, n_analyze_cycles=3):
    """Return (RMS_error_deg, c, analysis_dict) for given δ."""
    c_i = 2.0 * dlt * np.sqrt(I_A * (k_soft + m*g*L3))
    
    # Simulate multiple cycles
    t_cycles, theta_cycles, thetad_cycles, theta_ref_cycles = simulate_cycles(
        k_soft, c_i, f_target, Kp=Kp, Kd=Kd, theta0=theta0_mean, 
        ncycles=ncycles, n_per_cycle=400
    )
    
    # Analyze the last n_analyze_cycles for stability
    analysis = analyze_multiple_cycles(
        theta_cycles[-n_analyze_cycles:], 
        theta_ref_cycles[-n_analyze_cycles:],
        thetad_cycles[-n_analyze_cycles:],
        t_cycles[-n_analyze_cycles:],
        c_i
    )
    
    return analysis['rms_mean'], c_i, analysis

# Coarse sweep (25 δ values)
print("\n--- Starting coarse δ sweep ---")
delta_coarse = np.linspace(0.1, 0.8, 25)
rms_coarse = np.zeros_like(delta_coarse)
convergence_coarse = np.zeros_like(delta_coarse)

for i, dlt in enumerate(delta_coarse):
    rms_coarse[i], _, analysis = rms_for_delta(dlt)
    convergence_coarse[i] = analysis['convergence_ratio']
    if (i+1) % 5 == 0:
        print(f"  Completed {i+1}/25 coarse iterations...")

best_idx = int(np.argmin(rms_coarse))
delta_best_coarse = float(delta_coarse[best_idx])

# Fine sweep (15 δ around coarse best)
print("--- Starting fine δ sweep ---")
lo = max(0.1, delta_best_coarse - 0.05)
hi = min(0.8, delta_best_coarse + 0.05)
delta_fine = np.linspace(lo, hi, 15)
rms_fine = np.zeros_like(delta_fine)
c_fine   = np.zeros_like(delta_fine)
analysis_fine = []

for j, dlt in enumerate(delta_fine):
    rms_fine[j], c_fine[j], analysis = rms_for_delta(dlt)
    analysis_fine.append(analysis)
    if (j+1) % 5 == 0:
        print(f"  Completed {j+1}/15 fine iterations...")

j_best = int(np.argmin(rms_fine))
best_delta = float(delta_fine[j_best])
best_c     = float(c_fine[j_best])
best_rms   = float(rms_fine[j_best])
best_analysis = analysis_fine[j_best]

print(f"\n--- δ sweep results ---")
print(f"Best δ = {best_delta:.4f} → c = {best_c:.4f} N·m·s/rad")
print(f"RMS error: {best_rms:.3f}° ± {best_analysis['rms_std']:.3f}° (across {len(best_analysis['rms_per_cycle'])} cycles)")
print(f"Energy per cycle: {best_analysis['energy_mean']:.3f} ± {best_analysis['energy_std']:.3f} J")
print(f"Convergence ratio: {best_analysis['convergence_ratio']:.4f}")

# Plot RMS vs δ
plt.figure(figsize=(10,5))
plt.plot(delta_coarse, rms_coarse, 'o-', label="coarse")
plt.plot(delta_fine, rms_fine, 's-', label="fine")
plt.axvline(best_delta, color='r', ls='--', label=f"best δ={best_delta:.3f}")
plt.xlabel("Damping ratio δ"); plt.ylabel("RMS error [°]")
plt.title("RMS tracking error vs δ (coarse→fine sweep)")
plt.legend(); plt.grid(True, ls='--', alpha=0.6); plt.tight_layout(); plt.show()

# ----------------------------------------
# 8) Final high-res simulation at best δ with multiple cycle analysis
# ----------------------------------------
print("\n--- Final high-resolution simulation ---")
t_cycles, theta_cycles, thetad_cycles, theta_ref_cycles = simulate_cycles(
    k_soft, best_c, f_target, Kp=Kp, Kd=Kd, theta0=theta0_mean, ncycles=8, n_per_cycle=500
)

# Analyze all cycles
final_analysis = analyze_multiple_cycles(theta_cycles, theta_ref_cycles, thetad_cycles, t_cycles, best_c)

print(f"Final performance across {len(theta_cycles)} cycles:")
print(f"  RMS error: {final_analysis['rms_mean']:.3f}° ± {final_analysis['rms_std']:.3f}°")
print(f"  Energy per cycle: {final_analysis['energy_mean']:.3f} ± {final_analysis['energy_std']:.3f} J")
print(f"  Convergence: {final_analysis['convergence_ratio']:.4f}")

# Plot only the main comparison (last cycle vs experimental)
plt.figure(figsize=(10,6))
# Use the last cycle for the main plot
last_cycle_idx = -1
gait_percent = ((t_cycles[last_cycle_idx] - t_cycles[last_cycle_idx][0]) / T_cycle) * 100.0
theta_sim_deg = np.rad2deg(theta_cycles[last_cycle_idx])

plt.plot(t_s, ang_s, 'k', lw=2, label='Experimental gait (3.2 km/h)')
plt.plot(gait_percent, theta_sim_deg, 'g--', lw=2,
         label=f'Soft passive + PD (k={k_soft}, δ={best_delta:.3f})  RMS={final_analysis["rms_mean"]:.2f}°')

plt.xlabel("Gait cycle (%)"); plt.ylabel("Knee angle (°)")
plt.title("Experimental vs Simulated Knee Angle (3.2 km/h walking)")
plt.legend(); plt.grid(True, ls='--', alpha=0.6); plt.tight_layout(); plt.show()

# Final summary
print(f"\n{'='*60}")
print("PROSTHETIC TUNING RECOMMENDATION")
print(f"{'='*60}")
print(f"User height: {height:.1f} cm")
print(f"Walking speed: {speed_kmh} km/h (experimental data)")
print(f"Prosthetic mass: {m} kg")
print(f"Recommended stiffness: {k_soft:.1f} N·m/rad")
print(f"Recommended damping: {best_c:.3f} N·m·s/rad (δ={best_delta:.3f})")
print(f"Expected performance: {final_analysis['rms_mean']:.2f}° ± {final_analysis['rms_std']:.2f}° RMS error")
print(f"Energy dissipation: {final_analysis['energy_mean']:.3f} J/cycle")
print(f"Stability: {'GOOD' if final_analysis['convergence_ratio'] < 0.05 else 'MODERATE'}")
print(f"{'='*60}")