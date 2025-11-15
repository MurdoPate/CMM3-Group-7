# ================================================================
# Passive Mid-Swing Identification (k, c) via Grid Search + Secant
# Plots:
#   • Window highlight on mean shank angle (full gait, shaded passive region)
#   • Top grid fits + Secant-refined best (nonlinear model)
# Methods:
#   • RK4 ODE model
#   • Regression-based passive window (R²)
#   • Grid search + Secant refinement
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------- #
#           INPUT / SETUP         #
# ------------------------------- #

def GetHeight_cm() -> float:
    while True:
        try:
            h = float(input("Enter your height in cm: ").strip())
            if 80 < h < 250:
                return h
            print("Enter a value between 80 and 250.")
        except ValueError:
            print("Invalid number. Try e.g. 178.")


# ------------------------------- #
#       GEOMETRY & GAIT PARAMS    #
# ------------------------------- #

def CompGeo(height_cm: float, mass: float):
    lower_leg_cm = (height_cm - 123) / 1.24
    foot_cm      = 0.044 * height_cm + 19.286
    stride_m     = (0.415 * height_cm) / 100

    L1 = lower_leg_cm / 100
    L2 = foot_cm / 100

    den = 2 * (lower_leg_cm + foot_cm)
    y_bar = (lower_leg_cm**2) / den
    com_to_knee_cm = lower_leg_cm - y_bar
    L3 = com_to_knee_cm / 100

    I_A = (
        mass * (1.0 / (lower_leg_cm + foot_cm)) *
        ((1/3)*lower_leg_cm**3 + lower_leg_cm**2*foot_cm + (1/3)*foot_cm**3)
    ) / 1e4

    return dict(
        L1_cm=lower_leg_cm, L2_cm=foot_cm, L3_cm=com_to_knee_cm,
        L1=L1, L2=L2, L3=L3, I_A=I_A, stride=stride_m
    )


def gait_params(speed_kmh: float, stride_m: float):
    v = speed_kmh / 3.6
    f = v / stride_m
    T = 1 / f
    w = 2 * np.pi * f
    return dict(v=v, f=f, T=T, w=w)


# ------------------------------- #
#       DATA & PREPROCESSING      #
# ------------------------------- #

def CompShankGlob(df: pd.DataFrame, thigh_target_deg: float = 2.0):
    hip  = df[df.joint == 3].rename(columns={'angle': 'hip'})
    knee = df[df.joint == 2].rename(columns={'angle': 'knee'})
    keys = ['subject', 'condition', 'replication', 'leg', 'time']
    m = pd.merge(hip[keys+['hip']], knee[keys+['knee']], on=keys)

    out = []
    for _, d in m.groupby(['subject', 'replication', 'leg']):
        d = d.sort_values('time').copy()
        t = d['time'].to_numpy()
        hipj = d['hip'].to_numpy()
        mask = (t >= 0.10) & (t <= 0.30)
        tgt = np.deg2rad(thigh_target_deg)
        mean_hip = np.nanmean(hipj[mask]) if mask.any() else 0.0
        phi = tgt - mean_hip
        d['thigh'] = phi + d['hip']
        d['shank'] = d['thigh'] - d['knee']
        out.append(d)

    r = pd.concat(out, ignore_index=True)
    r['shank'] *= -1
    return r[['time', 'shank']]


def build_mean_curve(shank_df: pd.DataFrame, n: int = 1001, win: int = 9):
    df = shank_df.copy()
    df['t_round'] = (df['time'] * 1000).round().astype(int) / 1000
    g = df.groupby('t_round')['shank'].mean().sort_index()

    t_src, y_src = g.index.values, g.values
    good = np.isfinite(t_src) & np.isfinite(y_src)
    t_src, y_src = t_src[good], y_src[good]

    t_norm = np.linspace(0, 1, n)
    y = np.interp(t_norm, t_src, y_src)

    if win >= 3:
        if win % 2 == 0: win += 1
        pad = win // 2
        ypad = np.r_[y[pad:0:-1], y, y[-2:-pad-2:-1]]
        ker = np.ones(win) / win
        y = np.convolve(ypad, ker, mode='valid')[:len(y)]

    return t_norm, y


# ------------------------------- #
#       PASSIVE WINDOW (R²)       #
# ------------------------------- #

def deriv(y, x): return np.gradient(y, x)

def best_window(t, theta, wlen=0.15, search=(0.60, 0.95), step=0.005):
    best = None
    for s in np.arange(search[0], search[1]-wlen, step):
        e = s + wlen
        mask = (t >= s) & (t <= e) & np.isfinite(theta)
        if mask.sum() < 25: continue
        tt, th = t[mask], theta[mask]
        thd = deriv(th, tt)
        thdd = deriv(thd, tt)
        good = np.isfinite(th) & np.isfinite(thd) & np.isfinite(thdd)
        if good.sum() < 20: continue
        tt, th, thd, thdd = tt[good], th[good], thd[good], thdd[good]
        X = np.column_stack([thd, th])
        y = -thdd
        if np.var(y) < 1e-10: continue
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        yfit = X @ coef
        ss_res = np.sum((y - yfit)**2)
        ss_tot = np.sum((y - y.mean())**2)
        if ss_tot <= 1e-16: continue
        R2 = 1 - ss_res/ss_tot
        if not np.isfinite(R2): continue
        if (best is None) or (R2 > best['R2']): best = dict(s=s, e=e, R2=R2)
    return best


# ------------------------------- #
#         DYNAMICS (ODE)          #
# ------------------------------- #

def rk4_step(theta, omega, dt, I, c, k, m, g, L):
    def f1(theta, omega): return omega
    def f2(theta, omega): return -(c*omega + k*theta + m*g*L*np.sin(theta)) / I
    k1_th, k1_om = f1(theta, omega), f2(theta, omega)
    k2_th, k2_om = f1(theta+0.5*dt*k1_th, omega+0.5*dt*k1_om), f2(theta+0.5*dt*k1_th, omega+0.5*dt*k1_om)
    k3_th, k3_om = f1(theta+0.5*dt*k2_th, omega+0.5*dt*k2_om), f2(theta+0.5*dt*k2_th, omega+0.5*dt*k2_om)
    k4_th, k4_om = f1(theta+dt*k3_th, omega+dt*k3_om), f2(theta+dt*k3_th, omega+dt*k3_om)
    theta += (dt/6)*(k1_th+2*k2_th+2*k3_th+k4_th)
    omega += (dt/6)*(k1_om+2*k2_om+2*k3_om+k4_om)
    return theta, omega


def simulate_theta(t, theta0, omega0, I, c, k, m, g, L):
    theta, omega = theta0, omega0
    th = [theta]
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        theta, omega = rk4_step(theta, omega, dt, I, c, k, m, g, L)
        th.append(theta)
    return np.array(th)


# ------------------------------- #
#         COST & OPTIMISATION     #
# ------------------------------- #

def rmse(a, b): return np.sqrt(np.mean((a - b)**2))

def cost(time_s, theta_exp, theta0, omega0, I, m, g, L, k, c):
    th = simulate_theta(time_s, theta0, omega0, I, c, k, m, g, L)
    return rmse(th, theta_exp)

def dJ_dc(cf, k, c, h=1e-3):
    h = max(h, 1e-6)
    return (cf(k, c+h) - cf(k, c-h)) / (2*h)

def dJ_dk(cf, k, c, h=1e-3):
    h = max(h, 1e-6)
    return (cf(k+h, c) - cf(k-h, c)) / (2*h)

def secant(g, x1, x2, tol=1e-6, maxit=30, bounds=None):
    g1, g2 = g(x1), g(x2)
    for _ in range(maxit):
        d = g2 - g1
        if abs(d) < 1e-14: break
        x3 = x2 - g2 * (x2 - x1) / d
        if bounds: lo,hi=bounds; x3 = float(np.clip(x3, lo, hi))
        g3 = g(x3)
        if abs(g3) < tol: return x3
        x1,g1,x2,g2 = x2,g2,x3,g3
    return x2

def bracket(vals, idx):
    return float(vals[max(0, idx-1)]), float(vals[min(len(vals)-1, idx+1)])


# ------------------------------- #
#              PLOTTING           #
# ------------------------------- #

def plot_window_highlight(t_norm, theta_mean, s, e):
    plt.figure(figsize=(9, 4.8))
    plt.plot(t_norm*100, np.degrees(theta_mean), lw=2, label='Mean shank angle (deg)')
    plt.axvspan(s*100, e*100, color='C0', alpha=0.12, label='Passive window')
    plt.xlabel("Gait cycle (%)")
    plt.ylabel("Shank angle (deg)")
    plt.title("Passive window on mean shank-angle curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

def plot_top_overlays(t_win, theta_exp, k_vals, c_vals, J_grid,
                      k_star, c_star, sim_fn, top_n=5):
    flat = np.argpartition(J_grid.ravel(), top_n)[:top_n]
    pairs = np.column_stack(np.unravel_index(flat, J_grid.shape))
    pairs = sorted(pairs, key=lambda rc: J_grid[rc[0], rc[1]])

    plt.figure(figsize=(9, 5))
    plt.plot(t_win*100, np.degrees(theta_exp), lw=2, label='Experimental (deg)')

    for rank, (ik, jc) in enumerate(pairs, 1):
        k, c = k_vals[ik], c_vals[jc]
        th = sim_fn(k, c)
        plt.plot(t_win*100, np.degrees(th),
                 '--' if rank > 1 else '-', lw=2,
                 alpha=1 if rank == 1 else 0.75,
                 label=f"#{rank}: k={k:.3f}, c={c:.3f}, J={J_grid[ik,jc]:.4f}")

    th_star = sim_fn(k_star, c_star)
    plt.plot(t_win*100, np.degrees(th_star), ':', lw=2.5,
             label=f"Refined (k*={k_star:.3f}, c*={c_star:.3f})")

    plt.xlabel('Gait Cycle (%) within window')
    plt.ylabel('Shank Angle (deg)')
    plt.title('Top grid fits + Secant-refined best (nonlinear model)')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ------------------------------- #
#                MAIN             #
# ------------------------------- #

def main():
    g = 9.81
    mass = 4.0
    speed_kmh = 3.2

    height = GetHeight_cm()
    print(f"\nUsing height = {height:.1f} cm")

    geom = CompGeo(height, mass)
    gait = gait_params(speed_kmh, geom['stride'])

    print(f"Walking speed: {speed_kmh:.1f} km/h")
    print(f"L1={geom['L1_cm']:.2f} cm  L2={geom['L2_cm']:.2f} cm  "
          f"L3={geom['L3_cm']:.2f} cm ({geom['L3']:.3f} m)")
    print(f"I_A={geom['I_A']:.6f} kg·m²")
    print(f"Stride={geom['stride']:.3f} m  f={gait['f']:.3f} Hz  T={gait['T']:.3f} s")

    k0 = geom['I_A'] * gait['w']**2 - mass*g*geom['L3']
    print(f"Baseline k0: {k0:.3f} N·m/rad")

    # ✅ ---- UPDATED PART: Load CSV from GitHub instead of local path ----
    url = "https://raw.githubusercontent.com/MurdoPate/CMM3-Group-7/main/Gait_Biomechanics_Dataset.csv"
    df = pd.read_csv(url, encoding="latin1")
    # --------------------------------------------------------------------

    df = df[(df.condition == 1) & (df.leg == 2)].copy()

    shank = CompShankGlob(df)
    t, theta_mean = build_mean_curve(shank)

    win = best_window(t, theta_mean, wlen=0.15)
    if win is None:
        print("\nNo R² window found. Using 70–90%.")
        win = dict(s=0.70, e=0.90, R2=np.nan)

    s, e = win['s'], win['e']
    print(f"\nPassive region: {s*100:.1f}%–{e*100:.1f}% (R²={win['R2']:.3f})")

    mask = (t >= s) & (t <= e)
    t_win, theta_exp = t[mask], theta_mean[mask]
    time_s = t_win * gait['T']
    theta0 = float(theta_exp[0])
    omega0 = float(deriv(theta_exp, time_s)[0])

    def cf(k, c):
        return cost(time_s, theta_exp, theta0, omega0, geom['I_A'],
                    mass, g, geom['L3'], k, c)

    k_vals = np.linspace(max(0.05, 0.40*k0), 1.60*k0, 61)
    c_vals = np.linspace(0.0, 2.0, 81)

    J_grid = np.zeros((len(k_vals), len(c_vals)))
    best = (np.inf, None, None)
    for i, k in enumerate(k_vals):
        for j, c in enumerate(c_vals):
            J = cf(k, c)
            J_grid[i, j] = J
            if J < best[0]:
                best = (J, i, j)

    J0, i0, j0 = best
    k_best, c_best = float(k_vals[i0]), float(c_vals[j0])

    print("\nGrid best:")
    print(f"k0={k_best:.6f}, c0={c_best:.6f}, J0={J0:.6f}")

    k1, k2 = bracket(k_vals, i0)
    c1, c2 = bracket(c_vals, j0)

    k_star, c_star = k_best, c_best
    for _ in range(3):
        c_star = secant(lambda c: dJ_dc(cf, k_star, c),
                        c1, c2, tol=5e-7, bounds=(0.0, 2.0))
        c1, c2 = max(0, c_star-0.05), min(2.0, c_star+0.05)
        k_star = secant(lambda k: dJ_dk(cf, k, c_star),
                        k1, k2, tol=5e-7, bounds=(k_vals[0], k_vals[-1]))
        k1, k2 = max(k_vals[0], k_star-0.1), min(k_vals[-1], k_star+0.1)

    J_star = cf(k_star, c_star)

    print("\nRefined:")
    print(f"k*={k_star:.6f}, c*={c_star:.6f}, J*={J_star:.6f}")

    print("\n" + "="*66)
    print("                SUMMARY OF IDENTIFICATION RESULTS")
    print("="*66)
    print(f"Height:                    {height:.1f} cm")
    print(f"L1:                        {geom['L1_cm']:.2f} cm")
    print(f"L2:                        {geom['L2_cm']:.2f} cm")
    print(f"L3:                        {geom['L3_cm']:.2f} cm  ({geom['L3']:.3f} m)")
    print(f"I_A:                       {geom['I_A']:.6f} kg·m²")
    print("\nGait:")
    print(f"Speed:                     {gait['v']:.3f} m/s")
    print(f"Stride:                    {geom['stride']:.3f} m")
    print(f"f:                         {gait['f']:.3f} Hz")
    print(f"T:                         {gait['T']:.3f} s")
    print("\nPassive Window:")
    print(f"{s*100:.1f}% to {e*100:.1f}%")
    print(f"R²:                        {win['R2']:.4f}")
    print("\nParameters:")
    print(f"Baseline k0:               {k0:.4f} N·m/rad")
    print(f"Grid best:                 k={k_best:.4f}, c={c_best:.4f}, J0={J0:.6f}")
    print("Refined:")
    print(f"k*:                        {k_star:.4f} N·m/rad")
    print(f"c*:                        {c_star:.4f} N·m·s/rad")
    print(f"J*:                        {J_star:.6f} rad")
    print(f"J* (deg):                  {np.degrees(J_star):.4f} °")
    print("="*66 + "\n")

    def sim_fn(k, c):
        return simulate_theta(time_s, theta0, omega0, geom['I_A'], c, k, mass, g, geom['L3'])

    plot_window_highlight(t, theta_mean, s, e)
    plot_top_overlays(t_win, theta_exp, k_vals, c_vals, J_grid,
                      k_star, c_star, sim_fn, top_n=5)


if __name__ == "__main__":
    main()

