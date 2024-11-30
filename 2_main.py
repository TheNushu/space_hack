import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyIGRF
import matplotlib.animation as animation

class CubeSat:
    def __init__(self, mass=2.0, size=0.1):
        self.mass = mass
        self.size = size
        self.position = np.array([0., 0., 0.])
        self.velocity = np.array([0., 0., 0.])
        self.magnetic_moment_magnitude = 0.1
        self.orientation = np.array([1., 0., 0.])
        # Add rotation matrix to track full orientation
        self.rotation_matrix = np.eye(3)
        
    def align_with_magnetic_field(self, magnetic_field):
        if np.any(magnetic_field):
            # Calculate new z-axis (aligned with magnetic field)
            z_axis = magnetic_field / np.linalg.norm(magnetic_field)
            
            # Calculate new x-axis (perpendicular to z-axis)
            x_axis = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(x_axis, z_axis)) > 0.9:
                x_axis = np.array([0.0, 1.0, 0.0])
            x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            # Calculate new y-axis using cross product
            y_axis = np.cross(z_axis, x_axis)
            
            # Update rotation matrix
            self.rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            self.orientation = z_axis
        
    def get_magnetic_moment(self):
        return self.magnetic_moment_magnitude * self.orientation

class OrbitSimulation:
    def __init__(self):
        self.G = 6.67430e-11
        self.M_earth = 5.972e24
        self.R_earth = 6371000
        self.dt = 10.0
        self.altitude = 500000
        self.cubesat = CubeSat()
        self.initialize_orbit()
        self.positions = []
        self.magnetic_fields = []
        self.orientations = []
        self.times = []
        
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
        acc = self.gravitational_acceleration(self.cubesat.position)
        self.cubesat.velocity += acc * self.dt / 2
        self.cubesat.position += self.cubesat.velocity * self.dt
        acc_new = self.gravitational_acceleration(self.cubesat.position)
        self.cubesat.velocity += acc_new * self.dt / 2
        
        magnetic_field = self.get_magnetic_field(self.cubesat.position)
        self.cubesat.align_with_magnetic_field(magnetic_field)
        
        self.positions.append(self.cubesat.position.copy())
        self.magnetic_fields.append(magnetic_field)
        self.orientations.append(self.cubesat.orientation.copy())
        self.times.append(len(self.times) * self.dt)
        
    def run_simulation(self, duration):
        steps = int(duration / self.dt)
        for _ in range(steps):
            self.step()
            
    def animate_orbit(self):
        if not self.positions:
            self.run_simulation(6000)
            
        positions = np.array(self.positions)
        magnetic_fields = np.array(self.magnetic_fields)
        orientations = np.array(self.orientations)
        
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
        scale = self.altitude * 0.1  # For magnetic orientation
        ref_scale = scale * 0.5     # For reference axes
        
        # Create dummy artists for the first frame
        self._magnetic_quiver = None
        self._ref_axes = []  # List to store reference axes arrows
        
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
        ax1.set_title('Satellite Orbit and Magnetic Orientation')
        
        # Add legend entries
        ax1.plot([], [], 'r-', label='Satellite X axis')
        ax1.plot([], [], 'g-', label='Satellite Y axis')
        ax1.plot([], [], 'b-', label='Satellite Z axis')
        ax1.plot([], [], 'g-', linewidth=2, label='Magnetic Field')
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
            
            # Remove previous quivers
            if self._magnetic_quiver is not None:
                self._magnetic_quiver.remove()
            for arrow in self._ref_axes:
                arrow.remove()
            self._ref_axes = []
            
            # Update magnetic orientation quiver
            self._magnetic_quiver = ax1.quiver(positions[frame, 0], positions[frame, 1], positions[frame, 2],
                                             orientations[frame, 0], orientations[frame, 1], orientations[frame, 2],
                                             color='g', length=scale, linewidth=2)
            
            # Calculate reference frame vectors using rotation matrix
            x_axis = np.array([1, 0, 0])
            y_axis = np.array([0, 1, 0])
            z_axis = np.array([0, 0, 1])
            
            # Add reference axes with distinct colors and labels
            self._ref_axes.append(ax1.quiver(positions[frame, 0], positions[frame, 1], positions[frame, 2],
                                           x_axis[0] * ref_scale, x_axis[1] * ref_scale, x_axis[2] * ref_scale,
                                           color='red', linewidth=2))
            self._ref_axes.append(ax1.quiver(positions[frame, 0], positions[frame, 1], positions[frame, 2],
                                           y_axis[0] * ref_scale, y_axis[1] * ref_scale, y_axis[2] * ref_scale,
                                           color='green', linewidth=2))
            self._ref_axes.append(ax1.quiver(positions[frame, 0], positions[frame, 1], positions[frame, 2],
                                           z_axis[0] * ref_scale, z_axis[1] * ref_scale, z_axis[2] * ref_scale,
                                           color='blue', linewidth=2))
            
            # Update magnetic field plot
            field_line.set_data(time_hours[:frame], mag_field_magnitude[:frame] * 1e9)
            
            return (satellite, trail, field_line, self._magnetic_quiver, *self._ref_axes)
        
        anim = animation.FuncAnimation(fig, update, frames=len(positions),
                                     interval=1, blit=True)
        
        plt.tight_layout()
        plt.show()
        
        return anim

# Run simulation
sim = OrbitSimulation()
anim = sim.animate_orbit()