
__all__ = ['Config', 'device', 'dtype']

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

current_dir = Path(__file__).parent
data_dir = current_dir / "data"
data_dir.mkdir(exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

class Config:    
    def __init__(self):
        self.device = device
        self.dtype = dtype
        
        # Physical Parameters
        self.D = 1e-3              
        self.eta = 1.0            
        self.R0 = 3.0              
        self.f_D = 0.0            
        self.OD = 0.5            
        self.Qa = 1.0         
        self.Qb = 1.0            
        self.w = 0.7            
        
        # Numerical Parameters
        self.n1 = 40              
        self.n2 = 10               
        self.n3 = 40               
        self.T_final = 50.0       
        self.cfl = 0.1              
        self.convergence_tol = 1e-6  # Convergence tolerance
        self.ghostcell = 0           # Number of ghost cells
        
        # non-uniform grids
        self.grid_segments = [
            (0.0, 0.4),    # First segment
            (0.4, 0.6),    # Second segment
            (0.6, 1.0)     # Third segment
        ]
        
        # Boundary Conditions
        self.bc_type = 'dirichlet'  # Options: 'dirichlet', 'periodic'
        self.bc_value = 0.125       # Boundary value for Dirichlet BC
        
        # Initial Conditions
        self.initial_condition_type = 'uniform' # Options: 'uniform', 'analytical'
        self.initial_value = 0.125              # Initial value for uniform condition
        
    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        return {
            'D': torch.tensor(self.D, device=self.device, dtype=self.dtype),
            'eta': torch.tensor(self.eta, device=self.device, dtype=self.dtype),
            'R0': torch.tensor(self.R0, device=self.device, dtype=self.dtype),
            'f_D': torch.tensor(self.f_D, device=self.device, dtype=self.dtype),
            'OD': torch.tensor(self.OD, device=self.device, dtype=self.dtype),
            'Qa': torch.tensor(self.Qa, device=self.device, dtype=self.dtype),
            'Qb': torch.tensor(self.Qb, device=self.device, dtype=self.dtype),
            'w': torch.tensor(self.w, device=self.device, dtype=self.dtype),
            'bc_value': torch.tensor(self.bc_value, device=self.device, dtype=self.dtype),
            'initial_value': torch.tensor(self.initial_value, device=self.device, dtype=self.dtype),
        }
    
    def get_time_step(self, dd_min) -> torch.Tensor:
        dt = self.cfl * dd_min **2 / (6 * self.D)
        return dt.clone().detach().to(device=self.device, dtype=self.dtype)
    
    def create_non_uniform_grid(self):
        segments = self.grid_segments
        n1, n2, n3 = self.n1, self.n2, self.n3
        gc = self.ghostcell

        coords_1 = torch.linspace(segments[0][0], segments[0][1], n1+1, 
                                 device=self.device, dtype=self.dtype)
        coords_2 = torch.linspace(segments[1][0], segments[1][1], n2+1, 
                                 device=self.device, dtype=self.dtype)[1:]  
        coords_3 = torch.linspace(segments[2][0], segments[2][1], n3+1, 
                                 device=self.device, dtype=self.dtype)[1:]  
        coords = torch.cat([coords_1, coords_2, coords_3])

        if gc > 0:
            # Calculate spacing for ghost cells at the lower end
            lower_spacing = coords[1] - coords[0]
            lower_ghost = torch.linspace(
                coords[0] - gc * lower_spacing, 
                coords[0] - lower_spacing, 
                gc,
                device=self.device, dtype=self.dtype
            )
            
            # Calculate spacing for ghost cells at the upper end
            upper_spacing = coords[-1] - coords[-2]
            upper_ghost = torch.linspace(
                coords[-1] + upper_spacing, 
                coords[-1] + gc * upper_spacing, 
                gc,
                device=self.device, dtype=self.dtype
            )
            
            coords = torch.cat([lower_ghost, coords, upper_ghost])

        x_coords = coords - 0.5
        y_coords = coords - 0.5
        z_coords = coords.clone()
        
        grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')    
        coord_tensor = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        
        dx = grid_x[1:, :, :] - grid_x[:-1, :, :]
        dy = grid_y[:, 1:, :] - grid_y[:, :-1, :]
        dz = grid_z[:, :, 1:] - grid_z[:, :, :-1]    

        Mesh = {
        'grid': coord_tensor.cpu(),
        'grid_x': grid_x.cpu(),
        'grid_y': grid_y.cpu(),
        'grid_z': grid_z.cpu(), 
        'dx': dx.cpu(),
        'dy': dy.cpu(),
        'dz': dz.cpu()
        }
        torch.save(Mesh, data_dir/'Mesh.pt')
        
        return coord_tensor, grid_x, grid_y, grid_z, dx, dy, dz
    
    def __repr__(self):
        return f"""Config(
                    D={self.D}, eta={self.eta}, R0={self.R0}, f_D={self.f_D},
                    OD={self.OD}, Qa={self.Qa}, Qb={self.Qb}, w={self.w},
                    grid=({self.n1}, {self.n2}, {self.n3}), segments={self.grid_segments},
                    T_final={self.T_final}, ghostcell={self.ghostcell}
                )"""



if __name__ == "__main__":
    config = Config()

    print(config)
