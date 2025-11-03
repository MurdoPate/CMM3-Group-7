import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Anthropometric Scaling Functions ---
def estimate_segment_parameters(weight_kg, height_m, age):
    """
    Estimate shank/foot segment parameters based on body weight, height, and age.
    Uses Winter's biomechanics tables and regression equations.
    Returns: shank_mass, shank_length, l_cm (COM location), I (moment of inertia)
    """
    # Shank parameters (Winter 2009)
    shank_mass_frac = 0.0465  # Fraction of body weight
    shank_length_frac = 0.246  # Fraction of height
    shank_com_frac = 0.433     # COM location (fraction of shank length)

    # Calculate absolute values
    shank_mass = shank_mass_frac * weight_kg
    shank_length = shank_length_frac * height_m
    l_cm = shank_com_frac * shank_length

    # Moment of inertia (Yeadon 1989: I = 0.302 * m * l^2)
    I = 0.302 * shank_mass * (shank_length ** 2)

    # Age-adjusted gait frequency (Bohannon 1997)
    if age < 30:
        gait_freq = 1.1  # Hz (young adult)
    elif 30 <= age < 60:
        gait_freq = 1.0  # Hz (middle-aged)
    else:  # age >= 60
        gait_freq = 0.9 - 0.005 * (age - 60)  # Linear reduction for older adults

    return {
        "shank_mass": shank_mass,
        "shank_length": shank_length,
        "l_cm": l_cm,
        "I": I,
        "gait_freq": gait_freq
    }

def age_adjusted_theta_ref(phi, age):
    """
    Age-adjusted knee reference trajectory (Kerrigan et al. 2001).
    phi: Gait phase (0 to 1)
    age: Age in years
    """
    # Peak knee flexion (degrees) by age
    if age < 30:
        peak_flexion = 60.0
    elif 30 <= age < 60:
        peak_flexion = 55.0
    else:  # age >= 60
        peak_flexion = max(50.0 - 0.5 * (age - 60), 30.0)  # Linear reduction, min 30°

    # Smooth trajectory (sinusoidal)
    return np.radians(peak_flexion * np.sin(2 * np.pi * phi))

def dtheta_ref_dphi(phi, age):
    """
    Derivative of age-adjusted reference trajectory.
    """
    if age < 30:
        peak_flexion = 60.0
    elif 30 <= age < 60:
        peak_flexion = 55.0
    else:
        peak_flexion = max(50.0 - 0.5 * (age - 60), 30.0)

    return np.radians(peak_flexion * 2 * np.pi * np.cos(2 * np.pi * phi))

# --- ODE Model for Prosthetic Knee ---
def knee_rhs(t, y, params):
    """
    Right-hand side of the ODE for knee dynamics.
    y = [theta, theta_dot]
    """
    theta, theta_dot = y
    k, c, f, Kp, Kd, theta0 = params["k"], params["c"], params["f"], params["Kp"], params["Kd"], params["theta0"]
    shank_mass, l_cm, I = params["shank_mass"], params["l_cm"], params["I"]

    # Passive + gravity torque
    tau_joint = -k * (theta - theta0) - c * theta_dot
    tau_g = shank_mass * 9.81 * l_cm * np.sin(theta)

    # Reference tracking (PD control)
    phi = (f * t) % 1.0
    theta_des = age_adjusted_theta_ref(phi, params["age"])
    theta_des_dot = dtheta_ref_dphi(phi, params["age"]) * f
    tau_hip = Kp * (theta_des - theta) + Kd * (theta_des_dot - theta_dot)

    # Angular acceleration
    theta_ddot = (tau_hip - tau_g + tau_joint) / I
    return [theta_dot, theta_ddot]

# --- Simulate One Gait Cycle ---
def simulate_gait_cycle(weight_kg, height_m, age, k=100.0, c=10.0, Kp=5.0, Kd=0.5, theta0=0.0, n=2000):
    """
    Simulate one gait cycle for a user with given parameters.
    Returns: time, knee_angle, angular_velocity, reference_trajectory, params
    """
    # Estimate segment parameters
    seg_params = estimate_segment_parameters(weight_kg, height_m, age)
    seg_params.update({
        "k": k, "c": c, "Kp": Kp, "Kd": Kd, "theta0": theta0,
        "age": age, "f": seg_params["gait_freq"]
    })

    # Simulate ODE
    T = 1.0 / seg_params["f"]  # Gait cycle duration
    sol = solve_ivp(
        lambda t, y: knee_rhs(t, y, seg_params),
        [0.0, T], [0.0, 0.0],  # Initial conditions: theta=0, theta_dot=0
        method="RK45", rtol=1e-7, atol=1e-9, max_step=T/800,
        dense_output=True
    )

    # Interpolate results
    t_u = np.linspace(0, T, n, endpoint=False)
    theta_u = np.interp(t_u, sol.t, sol.y[0])
    thetad_u = np.interp(t_u, sol.t, sol.y[1])

    # Reference trajectory for comparison
    phi_t = (seg_params["f"] * t_u) % 1.0
    theta_ref_t = age_adjusted_theta_ref(phi_t, age)

    return t_u, theta_u, thetad_u, theta_ref_t, seg_params

# --- Energy Dissipation Calculation ---
def calculate_energy_dissipation(theta_dot, c, t):
    """
    Calculate energy dissipated by damping over one gait cycle.
    """
    from scipy.integrate import trapezoid
    return float(c * trapezoid(theta_dot**2, t))

# --- User Input and Simulation ---
def main():
    print("=== Prosthetic Knee Gait Simulator ===")
    weight_kg = float(input("Enter body weight (kg): "))
    height_m = float(input("Enter body height (m): "))
    age = int(input("Enter age (years): "))

    # Simulate gait cycle
    t_u, theta_u, thetad_u, theta_ref_t, params = simulate_gait_cycle(weight_kg, height_m, age)

    # Calculate energy dissipation
    E_damp = calculate_energy_dissipation(thetad_u, params["c"], t_u)

    # Print results
    print("\n--- Simulation Results ---")
    print(f"Shank mass: {params['shank_mass']:.3f} kg")
    print(f"Shank length: {params['shank_length']:.3f} m")
    print(f"Moment of inertia: {params['I']:.3f} kg·m²")
    print(f"Gait frequency: {params['f']:.2f} Hz")
    print(f"Energy dissipated: {E_damp:.3f} Joules")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_u, np.degrees(theta_u), label="Simulated Knee Angle")
    plt.plot(t_u, np.degrees(theta_ref_t), "--", label="Reference Trajectory")
    plt.ylabel("Knee Angle (degrees)")
    plt.legend()
    plt.grid(True)
    plt.title(f"Gait Simulation (Age {age}, Weight {weight_kg} kg, Height {height_m} m)")

    plt.subplot(2, 1, 2)
    plt.plot(t_u, np.degrees(thetad_u), label="Angular Velocity", color="orange")
    plt.ylabel("Angular Velocity (deg/s)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.show()

# --- Run the Simulation ---
if __name__ == "__main__":
    main()
