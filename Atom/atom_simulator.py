import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sp
from scipy.constants import *
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random

# Physical constants (some from scipy.constants, others defined)
ELECTRON_MASS = electron_mass  # kg
PROTON_MASS = proton_mass      # kg
NEUTRON_MASS = neutron_mass    # kg
ELEMENTARY_CHARGE = elementary_charge  # C
COULOMB_CONSTANT = 1 / (4 * pi * epsilon_0)  # N⋅m²/C²
BOHR_RADIUS = physical_constants['Bohr radius'][0]  # m
HARTREE_ENERGY = physical_constants['Hartree energy'][0]  # J
FINE_STRUCTURE = fine_structure  # dimensionless
HBAR = hbar  # J⋅s

@dataclass
class QuantumNumbers:
    """Represents the four quantum numbers of an electron"""
    n: int      # principal quantum number
    l: int      # angular momentum quantum number
    m_l: int    # magnetic quantum number
    m_s: float  # spin quantum number (-1/2 or +1/2)
    
    def __post_init__(self):
        if self.l >= self.n or self.l < 0:
            raise ValueError("l must be between 0 and n-1")
        if abs(self.m_l) > self.l:
            raise ValueError("m_l must be between -l and +l")
        if abs(self.m_s) != 0.5:
            raise ValueError("m_s must be ±1/2")

class Particle:
    """Base class for subatomic particles"""
    def __init__(self, mass: float, charge: float, position: np.ndarray = None):
        self.mass = mass
        self.charge = charge
        self.position = position if position is not None else np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.force = np.zeros(3)

class Electron(Particle):
    """Electron with quantum properties"""
    def __init__(self, quantum_numbers: QuantumNumbers, position: np.ndarray = None):
        super().__init__(ELECTRON_MASS, -ELEMENTARY_CHARGE, position)
        self.quantum_numbers = quantum_numbers
        self.orbital_energy = self.calculate_orbital_energy()
        self.orbital_radius = self.calculate_orbital_radius()
        
    def calculate_orbital_energy(self) -> float:
        """Calculate the energy of the electron in its orbital (Hydrogen-like)"""
        n = self.quantum_numbers.n
        # Simplified hydrogen-like energy levels
        return -13.6 * eV / (n**2)  # in Joules
        
    def calculate_orbital_radius(self) -> float:
        """Calculate the most probable radius for the electron"""
        n = self.quantum_numbers.n
        return n**2 * BOHR_RADIUS
        
    def wave_function_hydrogen(self, r: float, theta: float, phi: float, Z: int = 1) -> complex:
        """
        Calculate the hydrogen wave function ψ(r,θ,φ)
        This is a simplified version for lower quantum numbers
        """
        n, l, m = self.quantum_numbers.n, self.quantum_numbers.l, self.quantum_numbers.m_l
        
        # Normalization constant
        a0 = BOHR_RADIUS / Z  # effective Bohr radius
        
        # Radial part R_nl(r)
        rho = 2 * r / (n * a0)
        
        if n == 1 and l == 0:  # 1s orbital
            R_nl = 2 * (Z/a0)**(3/2) * np.exp(-rho/2)
        elif n == 2 and l == 0:  # 2s orbital
            R_nl = (1/2/np.sqrt(2)) * (Z/a0)**(3/2) * (2 - rho) * np.exp(-rho/2)
        elif n == 2 and l == 1:  # 2p orbital
            R_nl = (1/2/np.sqrt(6)) * (Z/a0)**(3/2) * rho * np.exp(-rho/2)
        else:
            # Simplified approximation for higher orbitals
            R_nl = np.exp(-rho/2) * (rho/n)**l
            
        # Angular part Y_l^m(θ,φ) - spherical harmonics
        Y_lm = sp.sph_harm(m, l, phi, theta)
        
        return R_nl * Y_lm
        
    def probability_density(self, r: float, theta: float, phi: float, Z: int = 1) -> float:
        """Calculate |ψ|² - probability density"""
        psi = self.wave_function_hydrogen(r, theta, phi, Z)
        return abs(psi)**2

class Proton(Particle):
    """Proton in the nucleus"""
    def __init__(self, position: np.ndarray = None):
        super().__init__(PROTON_MASS, ELEMENTARY_CHARGE, position)

class Neutron(Particle):
    """Neutron in the nucleus"""
    def __init__(self, position: np.ndarray = None):
        super().__init__(NEUTRON_MASS, 0, position)

class Nucleus:
    """Atomic nucleus containing protons and neutrons"""
    def __init__(self, atomic_number: int, mass_number: int):
        self.atomic_number = atomic_number  # Z
        self.mass_number = mass_number      # A
        self.neutron_number = mass_number - atomic_number  # N
        
        self.protons = [Proton() for _ in range(atomic_number)]
        self.neutrons = [Neutron() for _ in range(self.neutron_number)]
        
        # Nuclear radius approximation
        self.radius = 1.2e-15 * (mass_number**(1/3))  # meters
        
        # Arrange nucleons in a sphere
        self._arrange_nucleons()
        
    def _arrange_nucleons(self):
        """Arrange protons and neutrons in the nucleus"""
        all_nucleons = self.protons + self.neutrons
        
        for i, nucleon in enumerate(all_nucleons):
            # Random position within nuclear radius
            theta = random.uniform(0, 2*pi)
            phi = random.uniform(0, pi)
            r = random.uniform(0, self.radius)
            
            nucleon.position = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
            
    def total_charge(self) -> float:
        """Total charge of the nucleus"""
        return self.atomic_number * ELEMENTARY_CHARGE
        
    def total_mass(self) -> float:
        """Total mass of the nucleus"""
        return (self.atomic_number * PROTON_MASS + 
                self.neutron_number * NEUTRON_MASS)

class Atom:
    """Complete atom simulation with nucleus and electrons"""
    
    def __init__(self, element_symbol: str, atomic_number: int, mass_number: int):
        self.element_symbol = element_symbol
        self.atomic_number = atomic_number
        self.mass_number = mass_number
        
        # Create nucleus
        self.nucleus = Nucleus(atomic_number, mass_number)
        
        # Create electrons with proper electron configuration
        self.electrons = []
        self._populate_electrons()
        
        # Physical properties
        self.total_energy = 0.0
        self.ionization_energies = []
        
    def _populate_electrons(self):
        """Populate electrons according to Aufbau principle"""
        electron_config = self._get_electron_configuration()
        
        for orbital, num_electrons in electron_config.items():
            n, l = self._parse_orbital(orbital)
            
            # Fill orbitals according to Hund's rule
            for m_l in range(-l, l+1):
                if num_electrons <= 0:
                    break
                    
                # Add spin-up electron first
                if num_electrons > 0:
                    quantum_nums = QuantumNumbers(n, l, m_l, 0.5)
                    electron = Electron(quantum_nums)
                    self._set_electron_position(electron)
                    self.electrons.append(electron)
                    num_electrons -= 1
                    
                # Add spin-down electron if needed
                if num_electrons > 0:
                    quantum_nums = QuantumNumbers(n, l, m_l, -0.5)
                    electron = Electron(quantum_nums)
                    self._set_electron_position(electron)
                    self.electrons.append(electron)
                    num_electrons -= 1
                    
    def _get_electron_configuration(self) -> Dict[str, int]:
        """Get electron configuration for the atom"""
        # Simplified electron configurations for first 18 elements
        configs = {
            1: {"1s": 1},                                    # H
            2: {"1s": 2},                                    # He
            3: {"1s": 2, "2s": 1},                          # Li
            4: {"1s": 2, "2s": 2},                          # Be
            5: {"1s": 2, "2s": 2, "2p": 1},                # B
            6: {"1s": 2, "2s": 2, "2p": 2},                # C
            7: {"1s": 2, "2s": 2, "2p": 3},                # N
            8: {"1s": 2, "2s": 2, "2p": 4},                # O
            9: {"1s": 2, "2s": 2, "2p": 5},                # F
            10: {"1s": 2, "2s": 2, "2p": 6},               # Ne
            11: {"1s": 2, "2s": 2, "2p": 6, "3s": 1},      # Na
            12: {"1s": 2, "2s": 2, "2p": 6, "3s": 2},      # Mg
            13: {"1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 1},  # Al
            14: {"1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 2},  # Si
            15: {"1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 3},  # P
            16: {"1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 4},  # S
            17: {"1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 5},  # Cl
            18: {"1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 6},  # Ar
        }
        
        return configs.get(self.atomic_number, {"1s": min(2, self.atomic_number)})
        
    def _parse_orbital(self, orbital: str) -> Tuple[int, int]:
        """Parse orbital string like '2p' to get n and l values"""
        n = int(orbital[0])
        l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        l = l_map[orbital[1]]
        return n, l
        
    def _set_electron_position(self, electron: Electron):
        """Set initial position of electron based on its orbital"""
        # Use probabilistic positioning based on orbital
        n = electron.quantum_numbers.n
        l = electron.quantum_numbers.l
        
        # Sample from radial probability distribution
        r = self._sample_radial_distance(n, l)
        theta = random.uniform(0, pi)
        phi = random.uniform(0, 2*pi)
        
        electron.position = np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])
        
    def _sample_radial_distance(self, n: int, l: int) -> float:
        """Sample radial distance based on quantum numbers"""
        # Simplified sampling - use most probable radius with some variation
        r_max = n**2 * BOHR_RADIUS
        # Add randomness around the most probable radius
        return abs(random.gauss(r_max, r_max/4))
        
    def calculate_coulomb_force(self, particle1: Particle, particle2: Particle) -> np.ndarray:
        """Calculate Coulomb force between two particles"""
        r_vec = particle1.position - particle2.position
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag < 1e-20:  # Avoid division by zero
            return np.zeros(3)
            
        r_hat = r_vec / r_mag
        
        # F = k * q1 * q2 / r^2
        force_magnitude = (COULOMB_CONSTANT * particle1.charge * particle2.charge / 
                          (r_mag**2))
        
        return force_magnitude * r_hat
        
    def calculate_total_energy(self) -> float:
        """Calculate total energy of the atom"""
        kinetic_energy = 0.0
        potential_energy = 0.0
        
        # Kinetic energy of electrons
        for electron in self.electrons:
            v_squared = np.dot(electron.velocity, electron.velocity)
            kinetic_energy += 0.5 * electron.mass * v_squared
            
        # Potential energy from electron-nucleus interactions
        for electron in self.electrons:
            r = np.linalg.norm(electron.position)
            if r > 1e-20:
                potential_energy += (-COULOMB_CONSTANT * self.atomic_number * 
                                   ELEMENTARY_CHARGE**2 / r)
                
        # Electron-electron repulsion
        for i, e1 in enumerate(self.electrons):
            for j, e2 in enumerate(self.electrons[i+1:], i+1):
                r = np.linalg.norm(e1.position - e2.position)
                if r > 1e-20:
                    potential_energy += (COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / r)
                    
        self.total_energy = kinetic_energy + potential_energy
        return self.total_energy
        
    def simulate_time_step(self, dt: float):
        """Simulate one time step using classical mechanics"""
        # Reset forces
        for electron in self.electrons:
            electron.force = np.zeros(3)
            
        # Calculate forces on electrons
        for i, electron in enumerate(self.electrons):
            # Force from nucleus
            for proton in self.nucleus.protons:
                force = self.calculate_coulomb_force(electron, proton)
                electron.force -= force  # Attractive force
                
            # Force from other electrons
            for j, other_electron in enumerate(self.electrons):
                if i != j:
                    force = self.calculate_coulomb_force(electron, other_electron)
                    electron.force += force  # Repulsive force
                    
        # Update velocities and positions (Verlet integration)
        for electron in self.electrons:
            # a = F/m
            electron.acceleration = electron.force / electron.mass
            
            # Update velocity: v = v + a*dt
            electron.velocity += electron.acceleration * dt
            
            # Update position: x = x + v*dt
            electron.position += electron.velocity * dt
            
    def get_orbital_shapes(self, grid_size: int = 50) -> Dict[str, np.ndarray]:
        """Calculate orbital probability densities on a 3D grid"""
        orbital_shapes = {}
        
        # Create coordinate grid
        extent = 10 * BOHR_RADIUS  # 10 Bohr radii
        x = np.linspace(-extent, extent, grid_size)
        y = np.linspace(-extent, extent, grid_size)
        z = np.linspace(-extent, extent, grid_size)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Convert to spherical coordinates
        R = np.sqrt(X**2 + Y**2 + Z**2)
        THETA = np.arccos(Z / (R + 1e-10))
        PHI = np.arctan2(Y, X)
        
        # Calculate probability density for each unique orbital
        unique_orbitals = set()
        for electron in self.electrons:
            qn = electron.quantum_numbers
            orbital_key = f"{qn.n}{['s','p','d','f'][qn.l]}"
            unique_orbitals.add((orbital_key, qn.n, qn.l, qn.m_l))
            
        for orbital_key, n, l, m_l in unique_orbitals:
            dummy_qn = QuantumNumbers(n, l, m_l, 0.5)
            dummy_electron = Electron(dummy_qn)
            
            prob_density = np.zeros_like(R)
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        if R[i,j,k] > 1e-20:
                            prob_density[i,j,k] = dummy_electron.probability_density(
                                R[i,j,k], THETA[i,j,k], PHI[i,j,k], self.atomic_number)
                                
            orbital_shapes[orbital_key] = prob_density
            
        return orbital_shapes
        
    def visualize_atom(self, show_orbitals: bool = True, show_trajectories: bool = False):
        """Visualize the atom in 3D"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot nucleus
        ax.scatter([0], [0], [0], c='red', s=100, alpha=0.8, label='Nucleus')
        
        # Plot electrons
        electron_positions = np.array([e.position for e in self.electrons])
        ax.scatter(electron_positions[:, 0] / BOHR_RADIUS, 
                  electron_positions[:, 1] / BOHR_RADIUS,
                  electron_positions[:, 2] / BOHR_RADIUS, 
                  c='blue', s=50, alpha=0.6, label='Electrons')
        
        # Plot orbital boundaries (simplified as spheres)
        if show_orbitals:
            unique_n_values = set(e.quantum_numbers.n for e in self.electrons)
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']
            
            for i, n in enumerate(sorted(unique_n_values)):
                radius = n**2  # in Bohr radii
                u = np.linspace(0, 2 * np.pi, 50)
                v = np.linspace(0, np.pi, 50)
                x = radius * np.outer(np.cos(u), np.sin(v))
                y = radius * np.outer(np.sin(u), np.sin(v))
                z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, alpha=0.1, color=colors[i % len(colors)])
        
        ax.set_xlabel('X (Bohr radii)')
        ax.set_ylabel('Y (Bohr radii)')
        ax.set_zlabel('Z (Bohr radii)')
        ax.set_title(f'{self.element_symbol} Atom (Z={self.atomic_number})')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = 5  # Bohr radii
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        plt.tight_layout()
        plt.show()
        
    def print_atom_info(self):
        """Print detailed information about the atom"""
        print(f"=== {self.element_symbol} Atom Information ===")
        print(f"Atomic Number (Z): {self.atomic_number}")
        print(f"Mass Number (A): {self.mass_number}")
        print(f"Number of Protons: {self.nucleus.atomic_number}")
        print(f"Number of Neutrons: {self.nucleus.neutron_number}")
        print(f"Number of Electrons: {len(self.electrons)}")
        print(f"Nuclear Radius: {self.nucleus.radius:.2e} m")
        print(f"Nuclear Charge: {self.nucleus.total_charge()/ELEMENTARY_CHARGE:.1f} e")
        print(f"Nuclear Mass: {self.nucleus.total_mass():.2e} kg")
        print()
        
        print("Electron Configuration:")
        electron_config = {}
        for electron in self.electrons:
            qn = electron.quantum_numbers
            orbital = f"{qn.n}{['s','p','d','f'][qn.l]}"
            electron_config[orbital] = electron_config.get(orbital, 0) + 1
            
        config_str = " ".join([f"{orbital}^{count}" for orbital, count in electron_config.items()])
        print(f"  {config_str}")
        print()
        
        print("Electron Details:")
        for i, electron in enumerate(self.electrons):
            qn = electron.quantum_numbers
            print(f"  Electron {i+1}: n={qn.n}, l={qn.l}, m_l={qn.m_l}, m_s={qn.m_s}")
            print(f"    Orbital Energy: {electron.orbital_energy/eV:.2f} eV")
            print(f"    Orbital Radius: {electron.orbital_radius/BOHR_RADIUS:.2f} a₀")
            print(f"    Position: ({electron.position[0]/BOHR_RADIUS:.2f}, "
                  f"{electron.position[1]/BOHR_RADIUS:.2f}, "
                  f"{electron.position[2]/BOHR_RADIUS:.2f}) a₀")
        print()
        
        total_energy = self.calculate_total_energy()
        print(f"Total Energy: {total_energy/eV:.2f} eV")

# Example usage and demonstrations
if __name__ == "__main__":
    # Create a hydrogen atom
    print("Creating Hydrogen Atom...")
    hydrogen = Atom("H", 1, 1)
    hydrogen.print_atom_info()
    
    # Create a carbon atom
    print("\n" + "="*50)
    print("Creating Carbon Atom...")
    carbon = Atom("C", 6, 12)
    carbon.print_atom_info()
    
    # Demonstrate time evolution
    print("\n" + "="*50)
    print("Simulating atomic dynamics...")
    
    # Save initial positions
    initial_positions = [e.position.copy() for e in hydrogen.electrons]
    
    # Run simulation for a few time steps
    dt = 1e-18  # 1 attosecond
    for step in range(100):
        hydrogen.simulate_time_step(dt)
        
    # Show displacement
    final_positions = [e.position.copy() for e in hydrogen.electrons]
    for i, (initial, final) in enumerate(zip(initial_positions, final_positions)):
        displacement = np.linalg.norm(final - initial)
        print(f"Electron {i+1} displacement: {displacement/BOHR_RADIUS:.3f} a₀")
    
    # Visualize the atoms
    print("\nVisualizing atoms...")
    hydrogen.visualize_atom()
    carbon.visualize_atom()
    
    print("\nAtom simulation complete!") 