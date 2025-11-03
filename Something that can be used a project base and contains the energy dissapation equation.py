import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize
#Torque is equal to the 
#F_grf = (weight/2)*9.81
#c= any_value_assumed
#Calculate the reaction force from the ground at time t of the cycle. Assume
#foot is in contact with the ground for half a cycle for the rest F_grf is 
#constant at 0.
#During contact with the ground the reaction force is going to be modelled as
#a linear function with peak of F_grt at time t= 1/4 of cycle.
#Energy disspaption is equal to integral of instantaneosu power over the cycle.
#As c(damping coeeficient) is contstant.
#Energy = integral (c*(theta_dot(t)^ 2)) dt
# --- 1. Define Model Parameters and Inputs ---

def get_system_parameters():
    params = {}
    
    # Subject and Gait Parameters
    params['subject_mass'] = 70.0  # kg
    params['walking_speed'] = 1.2  # m/s
    
    # Estimated Gait Period (T_gait) for 1.2 m/s walking
    params['T_gait'] = 1.2  # seconds
    
    # Estimated Moment of Inertia (I) for shank + foot about the knee.
    # Literature values are often cited around 0.15 - 0.20 kg*m^2
    params['I'] = 0.17  # kg*m^2
    
    # Simulation settings
    params['num_cycles_to_settle'] = 10  # Run for N cycles to reach steady state
    
    return params

def create_torque_interpolator(mass, T_gait):
    
    #Creates and returns a continuous function for the Ground Reaction Force
    #torque (tau_GRF) over one gait cycle.
    
    # --- PLACEHOLDER DATA ---
    # This is a *normalized* knee torque curve for a walking gait (in Nm/kg).
    # You MUST replace this with data you source from a biomechanics paper.
    
    # Time as a percentage of the gait cycle
    time_percent = np.array([0, 10, 20, 30, 45, 60, 70, 85, 100])
    
    # Normalized torque (Nm / kg body mass)
    # This curve shows flexion (negative) then extension (positive)
    # during the stance phase (0-60%) and near-zero torque
    # during the swing phase (60-100%).
    torque_normalized_nm_per_kg = np.array([
        0.05, -0.4, -0.6, -0.3, 0.3, 0.1, -0.05, -0.1, 0.05
    ])
    
    # 1. Scale time from [0, 100] to [0, T_gait]
    time_actual = (time_percent / 100.0) * T_gait
    
    # 2. Scale torque from [Nm/kg] to [Nm] using subject's mass
    torque_actual = torque_normalized_nm_per_kg * mass
    
    # 3. Create the interpolation function
    # 'interp1d' returns a function that can be called with any 't'
    # 'bounds_error=False' and 'fill_value=0' means it returns 0
    # for any time outside the defined range (which shouldn't happen
    # if we use t % T_gait, but it's good practice).
    print("Creating GRF torque interpolator...")
    torque_func = interp1d(
        time_actual, 
        torque_actual,
        kind='cubic', # Smooth, cubic spline interpolation
        bounds_error=False,
        fill_value=0.0
    )
    
    # Return a "lambda" function that correctly handles periodic gait
    # (t % T_gait) ensures the torque repeats every cycle.
    return lambda t: torque_func(t % T_gait)

# --- 2. Define the ODE System ---

def ode_system(t, y, I, c, k, torque_func_t):
    """
    *** THIS IS THE 'ODE' STEP ***
    
    Defines the system of 1st-order differential equations.
    This is the function that `solve_ivp` will call.
    
    y is the state vector:
      y[0] = theta (angle, in radians)
      y[1] = theta_dot (angular velocity, in rad/s)
      
    Returns dy/dt = [ d(y[0])/dt, d(y[1])/dt ]
    """
    theta = y[0]
    theta_dot = y[1]
    
    # Get the external torque at time t
    tau_grf = torque_func_t(t)
    
    # Calculate angular acceleration (theta_double_dot)
    # From your equation: I*ddot + c*dot + k*theta = tau_GRF
    # Rearranged: I*ddot = tau_GRF - c*dot - k*theta
    theta_double_dot = (tau_grf - c * theta_dot - k * theta) / I
    
    # Return the derivatives
    return [theta_dot, theta_double_dot]

# --- 3. Define the Energy Calculation (Objective Function) ---

def calculate_energy_dissipation(ode_solution, c, t_start, t_end):
    """
    *** THIS IS THE 'INTEGRATION' STEP ***
    
    Calculates the total energy dissipated by the damper over one cycle.
    Energy = integral( c * (theta_dot(t))^2 ) dt
    """
    
    # Define the integrand function
    # ode_solution(t)[1] gives the interpolated value of theta_dot at time t
    integrand = lambda t: c * (ode_solution(t)[1])**2
    
    # Use quad to perform the numerical integration
    energy, error = quad(integrand, t_start, t_end)
    return energy

def objective_function(design_vars, params, torque_func):
    """
    This is the main function for the optimizer.
    It takes the design variables [k, c], runs a full simulation,
    and returns the energy dissipated, which the optimizer will try to minimize.
    """
    # Unpack the design variables
    k, c = design_vars
    
    # Get fixed parameters
    I = params['I']
    T_gait = params['T_gait']
    num_cycles = params['num_cycles_to_settle']
    
    # Define simulation time span
    t_start = 0
    t_end = T_gait * num_cycles
    t_span = [t_start, t_end]
    
    # Set initial conditions (e.g., 0 angle, 0 velocity)
    y0 = [0.0, 0.0]
    
    # --- Run the ODE Solver ---
    try:
        sol = solve_ivp(
            ode_system,
            t_span,
            y0,
            args=(I, c, k, torque_func),
            dense_output=True, # Essential for the integration step
            method='RK45'
        )
        
        # --- Calculate Energy for the *last* cycle ---
        # We use the last cycle to ensure the model is at steady state.
        t_start_cycle = T_gait * (num_cycles - 1)
        t_end_cycle = T_gait * num_cycles
        
        energy_dissipated = calculate_energy_dissipation(
            sol, c, t_start_cycle, t_end_cycle
        )
        
        # Handle potential non-physical results
        if not np.isfinite(energy_dissipated):
            return 1e10 # Return a large number if simulation failed
            
        print(f"Testing: k={k:8.2f}, c={c:8.2f} -> Energy={energy_dissipated:8.3f} J")
        return energy_dissipated
        
    except Exception as e:
        # print(f"Simulation failed for k={k}, c={c}: {e}")
        return 1e10 # Return a large number if simulation failed

# --- 4. Main Execution: Optimization and Plotting ---

def main():
    """
    Main function to orchestrate the optimization and plot the final result.
    """
    # 1. Load parameters
    params = get_system_parameters()
    
    # 2. Create the torque function (do this once)
    torque_func = create_torque_interpolator(
        params['subject_mass'], 
        params['T_gait']
    )
    
    # 3. --- Run the 'Root Finding' (Optimization) Step ---
    print("\n--- Starting Optimization ---")
    print("Finding optimal (k, c) to minimize energy dissipation...")
    
    # Initial guesses for [k, c]
    initial_guesses = [100.0, 10.0]
    
    # Bounds for k and c (must be positive)
    # (min_val, max_val)
    bounds = [(1.0, 500.0), (1.0, 100.0)]
    
    # We pass 'params' and 'torque_func' as extra arguments to the
    # objective function.
    result = minimize(
        objective_function,
        initial_guesses,
        args=(params, torque_func),
        method='L-BFGS-B', # A good method for bound constraints
        bounds=bounds,
        options={'disp': True, 'ftol': 1e-6}
    )
    
    # 4. --- Display Optimization Results ---
    print("\n--- Optimization Complete ---")
    if result.success:
        optimal_k, optimal_c = result.x
        min_energy = result.fun
        
        print(f"Optimal Stiffness (k): {optimal_k:.3f} Nm/rad")
        print(f"Optimal Damping (c): {optimal_c:.3f} Nms/rad")
        print(f"Minimum Energy Dissipated: {min_energy:.3f} Joules per cycle")
    else:
        print("Optimization failed to converge.")
        optimal_k, optimal_c = initial_guesses # Use guesses to plot
    
    # 5. --- Run and Plot the *Final, Optimal* Simulation ---
    print("\nRunning final simulation with optimal parameters...")
    
    # Get parameters for the final run
    I = params['I']
    T_gait = params['T_gait']
    num_cycles = params['num_cycles_to_settle']
    t_span = [0, T_gait * num_cycles]
    t_eval = np.linspace(t_span[0], t_span[1], num_cycles * 200)
    y0 = [0.0, 0.0]

    final_sol = solve_ivp(
        ode_system,
        t_span,
        y0,
        args=(I, optimal_c, optimal_k, torque_func),
        t_eval=t_eval,
        method='RK45'
    )
    
    # Plotting
    t = final_sol.t
    theta_deg = np.rad2deg(final_sol.y[0])
    theta_dot_deg_s = np.rad2deg(final_sol.y[1])
    
    # Also plot the input torque
    input_torque = [torque_func(ti) for ti in t]
    
    plt.figure(figsize=(12, 10))
    
    # Plot Knee Angle
    plt.subplot(3, 1, 1)
    plt.plot(t, theta_deg, label='Knee Angle (theta)')
    plt.title(f"Optimal Prosthetic Knee Simulation (k={optimal_k:.2f}, c={optimal_c:.2f})")
    plt.ylabel('Angle (degrees)')
    plt.grid(True)
    plt.legend()
    # Show only the last 3 cycles
    plt.xlim([T_gait * (num_cycles - 3), t.max()])

    # Plot Angular Velocity
    plt.subplot(3, 1, 2)
    plt.plot(t, theta_dot_deg_s, label='Angular Velocity (theta_dot)', color='orange')
    plt.ylabel('Angular Velocity (deg/s)')
    plt.grid(True)
    plt.legend()
    plt.xlim([T_gait * (num_cycles - 3), t.max()])
    
    # Plot Input Torque
    plt.subplot(3, 1, 3)
    plt.plot(t, input_torque, label='Input GRF Torque (tau_GRF)', color='red', linestyle='--')
    plt.ylabel('Torque (Nm)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()
    plt.xlim([T_gait * (num_cycles - 3), t.max()])
    
    plt.tight_layout()
    plt.show()

# --- Run the main function ---
if __name__ == "__main__":
    main()
