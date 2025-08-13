# 3D Quantum Diffusion Master Equation Simulation
This repository is to simulate the 3D Diffusion_Master_Equation with finite difference method.

## 1. Overview
This project simulates the 3D evolution of a quantum system using a finite difference approach with Runge-Kutta time integration. The code solves coupled partial differential equations describing spin dynamics with optical pumping and diffusion effects.

### 1.1 Key Features
- 3D non-uniform grid discretization
- Runge-Kutta 2nd order time integration
- Multiple boundary condition support (Dirichlet/periodic)
- Parallel GPU acceleration (CUDA supported)
- Comprehensive visualization tools
- Automatic simulation summary generation

### 1.2 File Structure
```
project_root/
├── main.py       	     # Main simulation driver
├── config.py          	 # Configuration parameters
├── spatial_discrete.py  # Spatial discretization implementation
├── plot.py              # Visualization tools
├── data/                # Output data directory (auto-created)
└── fig_YYYYMMDD_HHMMSS/ # Figure output directories (auto-created)
```

## 2. Usage
### 2.1 Requirements
- Python 3.8+
- Required packages:

  ```bash
  torch (PyTorch) >= 2.0
  numpy >= 1.20
  matplotlib >= 3.5
  pathlib (standard library)
  ```


### 2.2 Running the Simulation
```bash
python main.py
```

This will:

1. Generate a non-uniform 3D grid
2. Run the time evolution
3. Save results to `data/` directory
4. Automatically generate plots

### 2.3 Running Visualization Only
```bash
python plot.py
```
*Note:* Requires pre-existing data files in `data/` from a previous simulation.

### 2.4 Configuration
Modify `config.py` to adjust parameters.



### 2.5 Output Files
| File | Description |
|------|-------------|
| `rho_n.pt` | Final density field (torch tensor) |
| `P.pt` | Electric polarization rate (torch tensor) |
| `Mesh.pt` | Grid coordinates (torch tensor) |
| `simulation_summary_*.txt` | Simulation report |
| `fig_*/` | Directory containing visualization plots |

### 2.6 Visualization Features
The plotting script generates:

1. **Contour plots** of ρ₁ in three orthogonal planes:
   - YOZ plane (fixed X)
   - XOZ plane (fixed Y)
   - XOY plane (fixed Z)
   
2. **Component analysis**:


3. **Polarization visualization**:



You can change slice positions to modify visualization.


### 2.7 Performance Notes
- For GPU execution, set in `config.py`

- Memory requirements scale with grid size $(n_1 + n_2 + n_3)^3$



## 3. Contact
For questions or support, contact: 
 
**Xukeyu**  
Email: [xukeyu@csrc.ac.cn](mailto:xukeyu@csrc.ac.cn)  
Institution: CSRC 




