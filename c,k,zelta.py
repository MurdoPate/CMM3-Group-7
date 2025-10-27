import numpy as np
import math
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

# Assumptions
I = 0.20      # kg·m^2  inertia about knee
m = 4.0       # kg      shank+foot mass
l_cm = 0.20   # m       COM distance to knee
g = 9.81      # m/s^2
f_target = 1.0  # Hz    target gait frequency

# Damping ratios to evaluate
zeta_list = np.array([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])

# Reference knee trajectory θ_ref(φ) and dθ/dφ (φ in [0,1))
def theta_ref(phi):
    deg = 5 + 20*np.sin(2*np.pi*phi) + 15*np.sin(4*np.pi*(phi - 0.15))
    return np.deg2rad(deg)

def dtheta_dphi(phi):
    # derivative of the above w.r.t. phase φ, then deg→rad
    ddeg_dphi = (20*(2*np.pi)*np.cos(2*np.pi*phi)
                 + 15*(4*np.pi)*np.cos(4*np.pi*(phi - 0.15)))
    return np.deg2rad(ddeg_dphi)

# Root finding: solve k so that omega_n(k) = 2π f_target
def omega_n_from_k(k):
    k_eq = k - m*g*l_cm
    if k_eq <= 0:
        return float('nan')
    return math.sqrt(k_eq / I)

def solve_k_for_frequency(f):
    target = 2.0 * math.pi * f
    k_lo = m*g*l_cm * 1.000001
    k_hi = max(k_lo*2.0, 10.0)
    while omega_n_from_k(k_hi) < target:
        k_hi *= 2.0
        if k_hi > 1e8:
            raise RuntimeError("Failed to bracket k; check parameters.")
    return brentq(lambda kk: omega_n_from_k(kk) - target, k_lo, k_hi)

k_star = solve_k_for_frequency(f_target)
omega_n = omega_n_from_k(k_star)  # rad/s

# ODE model: I θ¨ + c θ˙ + k(θ - θ0) = τ_hip - τ_g
# τ_g = m g l_cm sin(θ); τ_hip = PD(θ_ref(t))
def knee_rhs(t, y, p):
    theta, theta_dot = y
    k, c, f, theta0, Kp, Kd = p["k"], p["c"], p["f"], p["theta0"], p["Kp"], p["Kd"]
    # passive + gravity
    tau_joint = -k*(theta - theta0) - c*theta_dot
    tau_g = m*g*l_cm*np.sin(theta)
    # reference tracking (light PD)
    phi = (f*t) % 1.0
    theta_des = theta_ref(phi)
    theta_des_dot = dtheta_dphi(phi) * f
    tau_hip = Kp*(theta_des - theta) + Kd*(theta_des_dot - theta_dot)
    theta_ddot = (tau_hip - tau_g + tau_joint) / I
    return [theta_dot, theta_ddot]

def simulate_one_cycle(k, c, f, Kp=5.0, Kd=0.5, theta0=0.0, n=2000):
    T = 1.0 / f
    p = dict(k=k, c=c, f=f, Kp=Kp, Kd=Kd, theta0=theta0)
    sol = solve_ivp(lambda t, y: knee_rhs(t, y, p),
                    [0.0, T], [0.0, 0.0],
                    method="RK45", rtol=1e-7, atol=1e-9, max_step=T/800)
    t_u = np.linspace(0, T, n, endpoint=False)
    theta_u   = np.interp(t_u, sol.t, sol.y[0])
    thetad_u  = np.interp(t_u, sol.t, sol.y[1])
    # reference for RMS
    phi_t = (f*t_u) % 1.0
    theta_ref_t = theta_ref(phi_t)
    return t_u, theta_u, thetad_u, theta_ref_t

def E_damp_from_traj(theta_dot, c, t):
    from scipy.integrate import trapezoid
    return float(c * trapezoid(theta_dot**2, t))

# Sweep over zeta → compute c, ODE simulation, E_damp, RMS
print("Assumptions: I=%.2f kg·m^2, m=%.1f kg, l_cm=%.2f m, f=%.2f Hz" % (I, m, l_cm, f_target))
print("Solved k by root-finding: k = %.4f N·m/rad" % k_star)
print("Natural frequency check:  ω_n = %.4f rad/s,  f_n = %.4f Hz\n" % (omega_n, omega_n/(2*np.pi)))

print("  zeta     c [N·m·s/rad]   E_damp [J/step]   RMS_error [deg]    omega_n [rad/s]")
for zeta in zeta_list:
    # c = 2 ζ sqrt(k_eq I)
    k_eq = k_star - m*g*l_cm
    c_val = 2.0 * zeta * math.sqrt(k_eq * I)
    t, th, thd, th_ref = simulate_one_cycle(k_star, c_val, f_target)
    E_damp = E_damp_from_traj(thd, c_val, t)
    rms_deg = float(np.rad2deg(np.sqrt(np.mean((th - th_ref)**2))))
    print(f" {zeta:>4.2f}      {c_val:>10.4f}        {E_damp:>10.4f}        {rms_deg:>10.3f}         {omega_n:>8.4f}")

