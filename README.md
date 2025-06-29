# Atom Simulation - High-Detail Physics Implementation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Physics](https://img.shields.io/badge/Physics-Quantum%20Mechanics-green.svg)](https://github.com/ArushCreater/Atom-Simulation)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive Python simulation of atoms with accurate quantum mechanical and classical physics implementations. This project provides educational insight into atomic physics while maintaining scientific rigor.

![Atom Visualization](https://via.placeholder.com/800x400/0066cc/ffffff?text=3D+Atom+Visualization)

## Features

### Quantum Mechanics
- **Four Quantum Numbers**: Each electron has proper n, l, m_l, and m_s quantum numbers with validation
- **Wave Functions**: Hydrogen-like wave functions with accurate radial and angular components
- **Spherical Harmonics**: Precise orbital shapes using scipy's spherical harmonics implementation
- **Energy Levels**: Correct energy calculations for different orbitals (-13.6 eV/n² for hydrogen-like atoms)
- **Electron Configuration**: Follows Aufbau principle, Pauli exclusion principle, and Hund's rules

### Atomic Structure
- **Complete Nucleus**: Protons and neutrons with correct masses and charges
- **Nuclear Physics**: Nuclear radius calculations using R = r₀A^(1/3) where r₀ = 1.2 fm
- **Physical Constants**: Uses scipy.constants for maximum precision
- **Coulomb Forces**: Accurate electrostatic interactions between all particles

### Classical Dynamics
- **Force Calculations**: Electron-nucleus attraction and electron-electron repulsion
- **Time Evolution**: Verlet integration for position and velocity updates
- **Energy Conservation**: Tracks kinetic and potential energy components
- **3D Motion**: Full three-dimensional particle dynamics

### Visualization & Analysis
- **3D Visualization**: Interactive matplotlib 3D plots of atomic structure
- **Orbital Boundaries**: Visual representation of electron orbitals
- **Probability Densities**: |ψ|² calculations for electron distributions
- **Real-time Dynamics**: Time-step simulation showing electron motion

**Created by [ArushCreater](https://github.com/ArushCreater)**

*"Simulating the fundamental building blocks of matter, one atom at a time."* 
