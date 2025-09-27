# 3D Quantum Diffusion Master Equation Simulation
This repository is to simulate the 3D Diffusion_Master_Equation with finite difference method.

## 1. Overview
This project simulates the 3D evolution of a quantum system using a finite difference approach with Runge-Kutta time integration with FOM solver. The code solves coupled partial differential equations describing spin dynamics with optical pumping and diffusion effects.

The ROM solver is an intrusive method. Hyperreduction with DEIM, NQE(nonlinear quadratic expansion).

## 2. Key Features
- 3D non-uniform grid discretization
- Runge-Kutta 2nd order time integration
- Parallel GPU acceleration (CUDA supported)
- Comprehensive visualization tools
- Automatic simulation summary generation