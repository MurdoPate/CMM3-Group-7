import numpy as np
import matplotlib.pyplot as plt

def calculate_angular_velocity(theta, time):
    """
    Calculate angular velocity (omega) from angle (theta) and time.

    Args:
        theta (np.ndarray): Knee flexion angle in radians.
        time (np.ndarray): Time points in seconds.

    Returns:
        np.ndarray: Angular velocity in rad/s.
    """
    omega = np.gradient(theta, time)
    return omega

def calculate_angular_acceleration(omega, time):
    """
    Calculate angular acceleration (alpha) from angular velocity (omega) and time.

    Args:
        omega (np.ndarray): Angular velocity in rad/s.
        time (np.ndarray): Time points in seconds.

    Returns:
        np.ndarray: Angular acceleration in rad/s².
    """
    alpha = np.gradient(omega, time)
    return alpha

def plot_kinematics(time, theta, omega, alpha):
    """
    Plot knee flexion angle, angular velocity, and angular acceleration over time.

    Args:
        time (np.ndarray): Time points in seconds.
        theta (np.ndarray): Knee flexion angle in degrees.
        omega (np.ndarray): Angular velocity in deg/s.
        alpha (np.ndarray): Angular acceleration in deg/s².
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    ax1.plot(time, theta, label='Flexion Angle (θ)')
    ax1.set_ylabel('Angle (deg)')
    ax1.legend()

    ax2.plot(time, omega, label='Angular Velocity (ω)', color='orange')
    ax2.set_ylabel('Velocity (deg/s)')
    ax2.legend()

    ax3.plot(time, alpha, label='Angular Acceleration (α)', color='green')
    ax3.set_ylabel('Acceleration (deg/s²)')
    ax3.set_xlabel('Time (s)')
    ax3.legend()

    plt.suptitle('Knee Kinematics During Gait Cycle')
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    time = np.linspace(0, 1, 100)  # Normalized gait cycle
    theta = 60 * np.sin(2 * np.pi * time)  # Example: 0-60° flexion (converted to radians in functions)
    omega = calculate_angular_velocity(theta * np.pi/180, time)  # Convert to radians
    alpha = calculate_angular_acceleration(omega, time)
    plot_kinematics(time, theta, omega * 180/np.pi, alpha * 180/np.pi)  # Convert back to degrees for plotting
