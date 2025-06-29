# 🔬 Atom Simulation - High-Detail Physics Implementation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Physics](https://img.shields.io/badge/Physics-Quantum%20Mechanics-green.svg)](https://github.com/ArushCreater/Atom-Simulation)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive Python simulation of atoms with accurate quantum mechanical and classical physics implementations. This project provides educational insight into atomic physics while maintaining scientific rigor.

![Atom Visualization](https://via.placeholder.com/800x400/0066cc/ffffff?text=3D+Atom+Visualization)

## 🚀 Features

### ⚛️ Quantum Mechanics
- **Four Quantum Numbers**: Each electron has proper n, l, m_l, and m_s quantum numbers with validation
- **Wave Functions**: Hydrogen-like wave functions with accurate radial and angular components
- **Spherical Harmonics**: Precise orbital shapes using scipy's spherical harmonics implementation
- **Energy Levels**: Correct energy calculations for different orbitals (-13.6 eV/n² for hydrogen-like atoms)
- **Electron Configuration**: Follows Aufbau principle, Pauli exclusion principle, and Hund's rules

### 🔬 Atomic Structure
- **Complete Nucleus**: Protons and neutrons with correct masses and charges
- **Nuclear Physics**: Nuclear radius calculations using R = r₀A^(1/3) where r₀ = 1.2 fm
- **Physical Constants**: Uses scipy.constants for maximum precision
- **Coulomb Forces**: Accurate electrostatic interactions between all particles

### ⚡ Classical Dynamics
- **Force Calculations**: Electron-nucleus attraction and electron-electron repulsion
- **Time Evolution**: Verlet integration for position and velocity updates
- **Energy Conservation**: Tracks kinetic and potential energy components
- **3D Motion**: Full three-dimensional particle dynamics

### 📊 Visualization & Analysis
- **3D Visualization**: Interactive matplotlib 3D plots of atomic structure
- **Orbital Boundaries**: Visual representation of electron orbitals
- **Probability Densities**: |ψ|² calculations for electron distributions
- **Real-time Dynamics**: Time-step simulation showing electron motion

## 🔧 Installation

### Prerequisites
```bash
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0
pip install scipy>=1.7.0
```

### Quick Start
```bash
git clone https://github.com/ArushCreater/Atom-Simulation.git
cd Atom-Simulation
python atom_simulator.py
```

## 📖 Usage

### Basic Atom Creation
```python
from atom_simulator import Atom

# Create atoms
hydrogen = Atom("H", 1, 1)
carbon = Atom("C", 6, 12)
oxygen = Atom("O", 8, 16)

# Print detailed information
hydrogen.print_atom_info()
```

### Time Evolution Simulation
```python
# Run dynamic simulation
dt = 1e-18  # 1 attosecond time step
initial_positions = [e.position.copy() for e in hydrogen.electrons]

for step in range(100):
    hydrogen.simulate_time_step(dt)

# Analyze results
final_positions = [e.position.copy() for e in hydrogen.electrons]
for i, (initial, final) in enumerate(zip(initial_positions, final_positions)):
    displacement = np.linalg.norm(final - initial)
    print(f"Electron {i+1} displacement: {displacement/BOHR_RADIUS:.3f} a₀")
```

### 3D Visualization
```python
# Visualize atoms with orbital boundaries
hydrogen.visualize_atom(show_orbitals=True)
carbon.visualize_atom(show_orbitals=True)

# Calculate orbital probability densities
orbital_shapes = carbon.get_orbital_shapes(grid_size=50)
```

## 🧪 Physics Accuracy

### Physical Constants Used
- **Electron mass**: 9.109 × 10⁻³¹ kg
- **Proton mass**: 1.673 × 10⁻²⁷ kg
- **Elementary charge**: 1.602 × 10⁻¹⁹ C
- **Bohr radius**: 5.292 × 10⁻¹¹ m
- **Coulomb constant**: 8.988 × 10⁹ N⋅m²/C²

### Quantum Mechanical Features
- **Wave Functions**: ψ(r,θ,φ) = R_nl(r) × Y_l^m(θ,φ)
- **Radial Functions**: Accurate R_nl(r) for s, p orbitals with Laguerre polynomials
- **Angular Functions**: Spherical harmonics Y_l^m(θ,φ) from scipy.special
- **Energy Eigenvalues**: E_n = -13.6 eV/n² for hydrogen-like atoms

### Supported Elements
Currently supports accurate electron configurations for elements 1-18:
- H, He (1s orbitals)
- Li through Ne (1s, 2s, 2p orbitals) 
- Na through Ar (1s, 2s, 2p, 3s, 3p orbitals)

## 🏗️ Code Structure

### Key Classes
- **`QuantumNumbers`**: Four quantum numbers with validation
- **`Particle`**: Base class for subatomic particles
- **`Electron`**: Quantum properties and wave functions
- **`Proton/Neutron`**: Nuclear particles
- **`Nucleus`**: Complete nuclear structure
- **`Atom`**: Full atomic simulation with dynamics

### Example Output
```
=== H Atom Information ===
Atomic Number (Z): 1
Mass Number (A): 1
Number of Electrons: 1
Nuclear Radius: 1.20e-15 m
Nuclear Charge: 1.0 e

Electron Configuration: 1s^1

Electron Details:
  Electron 1: n=1, l=0, m_l=0, m_s=0.5
    Orbital Energy: -13.60 eV
    Orbital Radius: 1.00 a₀
    Position: (-0.70, -0.89, -1.01) a₀

Total Energy: -17.89 eV
```

## 🎓 Educational Applications

This simulation is perfect for:
- **Physics Education**: Understanding atomic structure and quantum mechanics
- **Research**: Testing atomic physics concepts and calculations
- **Visualization**: Creating educational materials and presentations
- **Programming**: Learning scientific Python and computational physics

## 🔬 Technical Details

### Approximations and Limitations
- Uses hydrogen-like wave functions for multi-electron atoms
- Classical dynamics for time evolution (quantum effects approximated)
- Simplified nuclear structure (no nuclear forces)
- Limited to elements with Z ≤ 18

### Performance
- Optimized numpy operations for fast calculations
- Efficient 3D visualization with matplotlib
- Scalable time-step integration
- Memory-efficient particle representation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add more elements (transition metals, lanthanides)
- Implement relativistic effects
- Add molecular orbital calculations
- Improve visualization features
- Add unit tests

## 📚 References

- Griffiths, D.J. "Introduction to Quantum Mechanics"
- Atkins, P. "Physical Chemistry" 
- Bethe, H.A. "Quantum Mechanics of One- and Two-Electron Atoms"
- NIST Atomic Spectra Database

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with NumPy, SciPy, and Matplotlib
- Physical constants from CODATA/NIST
- Inspired by quantum mechanics textbooks and research

---

**Created by [ArushCreater](https://github.com/ArushCreater)**

*"Simulating the fundamental building blocks of matter, one atom at a time."* 