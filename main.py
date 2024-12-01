import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyIGRF
import matplotlib.animation as animation
import csv

def moment_of_inertia_tensor_cube(m, a):
    """Calculate moment of inertia tensor for a cube."""
    I_diagonal = (1 / 6) * m * a**2
    I_tensor = np.diag([I_diagonal, I_diagonal, I_diagonal])
    return I_tensor

def magnetic_torque(m_dipole, B_field):
    """Calculate magnetic torque on the satellite."""
    return np.cross(m_dipole, B_field)

def rotational_dynamics(I_tensor, omega, torque, dt):
    """Simulate rotational dynamics using Euler's equations."""
    I_inv = np.linalg.inv(I_tensor)
    omega_dot = np.dot(I_inv, torque - np.cross(omega, np.dot(I_tensor, omega)))
    return omega + omega_dot * dt

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]
    ])
    return R

def update_quaternion(q, omega, dt):
    """Update quaternion using angular velocity."""
    wx, wy, wz = omega
    Omega = 0.5 * np.array([
        [0, -wx, -wy, -wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0]
    ])
    q_dot = np.dot(Omega, q)
    q_new = q + q_dot * dt
    return q_new / np.linalg.norm(q_new)

class CubeSat:
    def __init__(self, mass=2.0, size=0.1):
        self.mass = mass
        self.size = size
        self.position = np.array([0., 0., 0.])
        self.velocity = np.array([0., 0., 0.])
        self.magnetic_moment_magnitude = 0.1
        
        # Rotational dynamics properties
        self.omega = np.array([0., 0., 0.])
        self.I_tensor = moment_of_inertia_tensor_cube(mass, size)
        self.quaternion = np.array([1., 0., 0., 0.])
        
    def get_magnetic_moment(self):
        R = quaternion_to_rotation_matrix(self.quaternion)
        body_moment = np.array([0., 0., self.magnetic_moment_magnitude])
        return np.dot(R, body_moment)
    
    def update_rotation(self, magnetic_field, dt):
        m_dipole = self.get_magnetic_moment()
        tau = magnetic_torque(m_dipole, magnetic_field)
        self.omega = rotational_dynamics(self.I_tensor, self.omega, tau, dt)
        self.quaternion = update_quaternion(self.quaternion, self.omega, dt)

class OrbitSimulation:
    def __init__(self):
        self.G = 6.67430e-11
        self.M_earth = 5.972e24
        self.R_earth = 6371000
        self.dt = 10.0
        self.altitude = 500000
        self.cubesat = CubeSat()
        self.positions = []
        self.magnetic_fields = []
        self.quaternions = []
        self.omegas = []
        self.times = []
        self.initialize_orbit()
        
    def initialize_orbit(self):
        r = self.R_earth + self.altitude
        v_orbital = np.sqrt(self.G * self.M_earth / r)
        self.cubesat.position = np.array([r, 0., 0.])
        self.cubesat.velocity = np.array([0., v_orbital, 0.])
        
    def gravitational_acceleration(self, position):
        r = norm(position)
        return -self.G * self.M_earth * position / (r**3)
        
    def get_magnetic_field(self, position):
        r = norm(position)
        lat = np.arcsin(position[2] / r)
        lon = np.arctan2(position[1], position[0])
        lat_deg = np.degrees(lat)
        lon_deg = np.degrees(lon)
        alt_km = (r - self.R_earth) / 1000
        
        try:
            mag_field = pyIGRF.igrf_value(lat_deg, lon_deg, alt_km, 2024)
            return np.array([mag_field[3], mag_field[4], mag_field[5]]) * 1e-9
        except:
            return np.zeros(3)
        
    def step(self):
        # Orbital dynamics
        acc = self.gravitational_acceleration(self.cubesat.position)
        self.cubesat.velocity += acc * self.dt / 2
        self.cubesat.position += self.cubesat.velocity * self.dt
        acc_new = self.gravitational_acceleration(self.cubesat.position)
        self.cubesat.velocity += acc_new * self.dt / 2
        
        # Rotational dynamics
        magnetic_field = self.get_magnetic_field(self.cubesat.position)
        self.cubesat.update_rotation(magnetic_field, self.dt)
        
        # Store data
        self.positions.append(self.cubesat.position.copy())
        self.magnetic_fields.append(magnetic_field)
        self.quaternions.append(self.cubesat.quaternion.copy())
        self.omegas.append(self.cubesat.omega.copy())
        self.times.append(len(self.times) * self.dt)
        
    def run_simulation(self, duration):
        steps = int(duration / self.dt)
        for _ in range(steps):
            self.step()
            
    def save_data(self, filename):
        rows = []
        for i in range(len(self.times)):
            row = {
                'time': self.times[i],
                'pos_x': self.positions[i][0],
                'pos_y': self.positions[i][1],
                'pos_z': self.positions[i][2],
                'mag_x': self.magnetic_fields[i][0],
                'mag_y': self.magnetic_fields[i][1],
                'mag_z': self.magnetic_fields[i][2],
                'quat_w': self.quaternions[i][0],
                'quat_x': self.quaternions[i][1],
                'quat_y': self.quaternions[i][2],
                'quat_z': self.quaternions[i][3],
                'omega_x': self.omegas[i][0],
                'omega_y': self.omegas[i][1],
                'omega_z': self.omegas[i][2]
            }
            rows.append(row)

        with open(f"{filename}.csv", 'w', newline='') as file:
            fieldnames = ['time', 
                         'pos_x', 'pos_y', 'pos_z',
                         'mag_x', 'mag_y', 'mag_z',
                         'quat_w', 'quat_x', 'quat_y', 'quat_z',
                         'omega_x', 'omega_y', 'omega_z']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            
    def animate_orbit(self):
        if not self.positions:
            print("No simulation data available. Running simulation first...")
            self.run_simulation(6000)
            
        positions = np.array(self.positions)
        magnetic_fields = np.array(self.magnetic_fields)
        quaternions = np.array(self.quaternions)
        
        fig = plt.figure(figsize=(15, 7))
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot Earth
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.R_earth * np.outer(np.cos(u), np.sin(v))
        y = self.R_earth * np.outer(np.sin(u), np.sin(v))
        z = self.R_earth * np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x, y, z, color='b', alpha=0.1)
        
        # Initial plots
        satellite, = ax1.plot([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 
                            'ro', markersize=10, label='CubeSat')
        trail, = ax1.plot([], [], [], 'r-', alpha=0.5, label='Orbit Trail')
        
        # Scale factors
        scale = self.altitude * 0.1
        ref_scale = scale * 0.5
        
        # Create dummy artists for the first frame
        self._ref_axes = []
        
        # Magnetic field plot
        ax2 = fig.add_subplot(122)
        mag_field_magnitude = np.linalg.norm(magnetic_fields, axis=1)
        time_hours = np.array(self.times) / 3600
        field_line, = ax2.plot([], [], 'b-')
        
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Magnetic Field Magnitude (nT)')
        ax2.set_title('Magnetic Field Strength vs Time')
        ax2.grid(True)
        ax2.set_xlim(0, time_hours[-1])
        ax2.set_ylim(0, max(mag_field_magnitude * 1e9) * 1.1)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Satellite Orbit and Orientation')
        
        # Add legend entries
        ax1.plot([], [], 'r-', label='Body X axis')
        ax1.plot([], [], 'g-', label='Body Y axis')
        ax1.plot([], [], 'b-', label='Body Z axis')
        ax1.legend()
        
        # Set equal aspect ratio
        max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                            positions[:, 1].max()-positions[:, 1].min(),
                            positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
        mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        
        trail_length = 100
        
        def update(frame):
            # Update satellite position
            satellite.set_data([positions[frame, 0]], [positions[frame, 1]])
            satellite.set_3d_properties([positions[frame, 2]])
            
            # Update trail
            start = max(0, frame - trail_length)
            trail.set_data(positions[start:frame, 0], positions[start:frame, 1])
            trail.set_3d_properties(positions[start:frame, 2])
            
            # Remove previous arrows
            for arrow in self._ref_axes:
                arrow.remove()
            self._ref_axes = []
            
            # Get rotation matrix from quaternion
            R = quaternion_to_rotation_matrix(quaternions[frame])
            
            # Draw body axes
            colors = ['red', 'green', 'blue']
            for i, axis in enumerate([R[:, 0], R[:, 1], R[:, 2]]):
                self._ref_axes.append(ax1.quiver(positions[frame, 0],
                                               positions[frame, 1],
                                               positions[frame, 2],
                                               axis[0] * ref_scale,
                                               axis[1] * ref_scale,
                                               axis[2] * ref_scale,
                                               color=colors[i],
                                               linewidth=2))
            
            # Update magnetic field plot
            field_line.set_data(time_hours[:frame], mag_field_magnitude[:frame] * 1e9)
            
            return (satellite, trail, field_line, *self._ref_axes)
        
        anim = animation.FuncAnimation(fig, update, frames=len(positions),
                                     interval=1, blit=True)
        
        plt.tight_layout()
        plt.show()
        
        return anim

if __name__ == "__main__":
    # Create simulation instance
    sim = OrbitSimulation()
    
    # Run simulation
    print("Running simulation...")
    sim.run_simulation(6000)  # 6000 seconds duration
    
    # Save data
    print("Saving simulation data...")
    sim.save_data("orbit_simulation_data")
    
    # Create animation
    print("Creating animation...")
    anim = sim.animate_orbit()
