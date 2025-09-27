"""
Optimized spatial discretization module for 3D diffusion simulation.
Features:
- Externalized parameter configuration
- Precomputation optimizations
- Clean code structure
"""

import torch
from typing import Dict, Tuple, Optional
from config import Config, device, dtype


class Spatial_fun:
    def __init__(self, config: Config, grid: torch.Tensor, grid_x: torch.Tensor, 
                 grid_y: torch.Tensor, grid_z: torch.Tensor, 
                 dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor, 
                 Nx: int, Ny: int, Nz: int):
        """
        Initialize spatial discretization object.
        
        Args:
            config: Configuration object
            grid: Mesh coordinates tensor (Nx, Ny, Nz, 3)
            grid_x, grid_y, grid_z: Mesh coordinates in each direction
            dx, dy, dz: Grid spacings
            Nx, Ny, Nz: Grid dimensions
        """
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        
        self.grid = grid.to(self.device)
        self.grid_x = grid_x.to(self.device)
        self.grid_y = grid_y.to(self.device)
        self.grid_z = grid_z.to(self.device)
        
        self.dx = self._ensure_4d(dx.to(self.device))
        self.dy = self._ensure_4d(dy.to(self.device))
        self.dz = self._ensure_4d(dz.to(self.device))
        
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        
        self.params = config.to_tensor_dict()
        self._setup_matrices()
        self._setup_coefficient()
        
    def _ensure_4d(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) == 3:
            return tensor.unsqueeze(-1)
        return tensor
    
    def _setup_matrices(self):
        # A_SD matrix (Spin Dephasing)
        self.A_SD = 0.0625 * torch.tensor([
            [8, -2, 0, 0, 0, 0, 0, -6],
            [-2, 11, -3, 0, 0, 0, -3, -3],
            [0, -3, 12, -3, 0, -1, -4, -1],
            [0, 0, -3, 11, -2, -3, -3, 0],
            [0, 0, 0, -2, 8, -6, 0, 0],
            [0, 0, -1, -3, -6, 11, -1, 0],
            [0, -3, -4, -3, 0, -1, 12, -1],
            [-6, -3, -1, 0, 0, 0, -1, 11]
        ], device=self.device, dtype=self.dtype)
        
        # A_FD matrix (Energy Level Relaxation)
        self.A_FD = torch.tensor([
            [2, -2, 0, 0, 0, 0, 0, 0],
            [-2, 5, -3, 0, 0, 0, 0, 0],
            [0, -3, 6, -3, 0, 0, 0, 0],
            [0, 0, -3, 5, -2, 0, 0, 0],
            [0, 0, 0, -2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, 0],
            [0, 0, 0, 0, 0, -1, 2, -1],
            [0, 0, 0, 0, 0, 0, -1, 1]
        ], device=self.device, dtype=self.dtype)
        
        # A_op matrix (Optical Pumping)
        A_op_base = torch.tensor([
            [0, -0.25, 0, 0, 0, 0, 0, -0.75],
            [0, 0.4375, -0.375, 0, 0, 0, -0.375, -0.1875],
            [0, 0, 0.75, -0.375, 0, -0.125, -0.25, 0],
            [0, 0, 0, 0.9375, -0.25, -0.1875, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, -0.1875, -0.75, 0.4375, 0, 0],
            [0, 0, -0.25, -0.375, 0, -0.125, 0.75, 0],
            [0, -0.1875, -0.125, 0, 0, 0, -0.125, 0.9375]
        ], device=self.device, dtype=self.dtype)
        
        # Q_ab matrix (Quantum Efficiency)
        Q_ab = torch.diag(torch.cat([
            self.params['Qa'].repeat(5),
            self.params['Qb'].repeat(3)
        ]))
        
        self.A_op = A_op_base @ Q_ab
        
        # A_SE matrix (Spin Exchange)
        self.A_SE = 0.125 * torch.tensor([
            [8, 2, 0, 0, 0, 0, 0, 6],
            [-2, 4, 3, 0, 0, 0, 3, 0],
            [0, -3, 0, 3, 0, 1, 0, -1],
            [0, 0, -3, -4, 2, 0, -3, 0],
            [0, 0, 0, -2, -8, -6, 0, 0],
            [0, 0, -1, 0, 6, 4, -1, 0],
            [0, -3, 0, 3, 0, 1, 0, -1],
            [-6, 0, 1, 0, 0, 0, 1, -4]
        ], device=self.device, dtype=self.dtype)
        
        # S_z vector (Spin projection operator)
        self.S_z = torch.tensor(
            [0.5, 0.25, 0, -0.25, -0.5, 0.25, 0, -0.25],
            device=self.device, dtype=self.dtype
        )
        
        # Combined G0 matrix
        self.G0_matrix = ((1 + self.params['eta']) * self.A_SD + 
                          self.params['f_D'] * self.A_FD)
        
    def _setup_coefficient(self):
        # x direction
        dx_im1 = self.dx[:-1, 1:-1, 1:-1, :]  # dx_{i-1/2}
        dx_ip1 = self.dx[1:, 1:-1, 1:-1, :]   # dx_{i+1/2}        
        
        self.alpha_x = 2.0 / (dx_im1 * (dx_im1 + dx_ip1))    # Calculate weight coefficient 
        self.beta_x = -2.0 / (dx_im1 * dx_ip1)
        self.gamma_x = 2.0 / (dx_ip1 * (dx_im1 + dx_ip1))

        # y direction
        dy_jm1 = self.dy[1:-1, :-1, 1:-1, :]  # dy_{j-1/2}
        dy_jp1 = self.dy[1:-1, 1:, 1:-1, :]   # dy_{j+1/2}
        
        self.alpha_y = 2.0 / (dy_jm1 * (dy_jm1 + dy_jp1))
        self.beta_y = -2.0 / (dy_jm1 * dy_jp1)
        self.gamma_y = 2.0 / (dy_jp1 * (dy_jm1 + dy_jp1))

        # z direction
        dz_km1 = self.dz[1:-1, 1:-1, :-1, :]  # dz_{k-1/2}
        dz_kp1 = self.dz[1:-1, 1:-1, 1:, :]   # dz_{k+1/2}
        
        self.alpha_z = 2.0 / (dz_km1 * (dz_km1 + dz_kp1))
        self.beta_z = -2.0 / (dz_km1 * dz_kp1)
        self.gamma_z = 2.0 / (dz_kp1 * (dz_km1 + dz_kp1))
    
        # Precompute spatial profile for optical pumping
        x, y, z = self.grid_x, self.grid_y, self.grid_z
        w = self.params['w']
        OD = self.params['OD']
        
        self.xi_r = torch.exp(-(x**2 + y**2)/(2*w**2)) * torch.exp(-OD * z)
        self.xi_r = self.xi_r.unsqueeze(-1)  # (Nx, Ny, Nz, 1)
    
    def _setup_boundary_conditions(self, rho: torch.Tensor, gc: torch.Tensor,
                                 bc_type: Optional[str] = None, 
                                 bc_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        if bc_type is None:
            bc_type = self.config.bc_type
        if bc_value is None:
            bc_value = self.params['bc_value']
        
        if bc_type == 'dirichlet':
            # x direction 
            rho[0, :, :, :] = bc_value
            rho[-1, :, :, :] = bc_value

            # y direction
            rho[:, 0, :, :] = bc_value
            rho[:, -1, :, :] = bc_value

            # z direction
            rho[:, :, 0, :] = bc_value
            rho[:, :, -1, :] = bc_value
            
        elif bc_type == 'periodic':
            # Apply periodic boundary conditions using ghost cells
            rho[:gc, :, :, :] = rho[-2*gc:-gc, :, :, :]  # Front = back interior
            rho[-gc:, :, :, :] = rho[gc:2*gc, :, :, :]   # Back = front interior

            rho[:, :gc, :, :] = rho[:, -2*gc:-gc, :, :]  # Left = right interior
            rho[:, -gc:, :, :] = rho[:, gc:2*gc, :, :]   # Right = left interior
            
            rho[:, :, :gc, :] = rho[:, :, -2*gc:-gc, :]  # Bottom = top interior
            rho[:, :, -gc:, :] = rho[:, :, gc:2*gc, :]   # Top = bottom interior

        elif bc_type == 'neumann':
            pass
        else:
            raise ValueError(f"Unsupported boundary condition type: {bc_type}")
        
        return rho
    
    def _setup_initial_condition(self, rho: torch.Tensor) -> torch.Tensor:
        if self.config.initial_condition_type == 'uniform':
            rho[:, :, :, :] = self.params['initial_value']
        elif self.config.initial_condition_type == 'analytical':
            pass
        
        return rho
    
    def _setup_steady_state_solution(self) -> torch.Tensor:
        x = 0.0
        y = 0.0
        z = 0.5
        xi_r0 = torch.exp(-(x**2 + y**2)/(2*self.params['w']**2)) * torch.exp(-self.params['OD'] * z)
        xi_r0 = xi_r0.clone().detach().to(dtype=torch.float64, device=self.device)
        R = self.params['R0'] * xi_r0
        P = R / (R + 1)
        denominator = 8 * (P**2 + 1)
        
        rho_0 = torch.tensor([
            (P + 1)**4 / denominator,
            -(P - 1) * (P + 1)**3 / denominator,
            (P**2 - 1)**2 / denominator,
            -(P - 1)**3 * (P + 1) / denominator,
            (P - 1)**4 / denominator,
            -(P - 1)**3 * (P + 1) / denominator,
            (P**2 - 1)**2 / denominator,
            -(P - 1) * (P + 1)**3 / denominator
        ], device=self.device, dtype=self.dtype)
        
        return rho_0
    
    def rhs_diffusion(self, rho: torch.Tensor) -> torch.Tensor:
        # Calculating second derivative
        d2rho_dx2 = (self.alpha_x * rho[:-2, 1:-1, 1:-1, :] + 
                     self.beta_x * rho[1:-1, 1:-1, 1:-1, :] + 
                     self.gamma_x * rho[2:, 1:-1, 1:-1, :])
        
        d2rho_dy2 = (self.alpha_y * rho[1:-1, :-2, 1:-1, :] + 
                     self.beta_y * rho[1:-1, 1:-1, 1:-1, :] + 
                     self.gamma_y * rho[1:-1, 2:, 1:-1, :])
        
        d2rho_dz2 = (self.alpha_z * rho[1:-1, 1:-1, :-2, :] + 
                     self.beta_z * rho[1:-1, 1:-1, 1:-1, :] + 
                     self.gamma_z * rho[1:-1, 1:-1, 2:, :])
        
        nabla2 = d2rho_dx2 + d2rho_dy2 + d2rho_dz2
        return self.params['D'] * nabla2
    
    def rhs_G0(self, rho: torch.Tensor) -> torch.Tensor:
        rhs_G0 = torch.einsum('ij,klmj->klmi', self.G0_matrix, rho)
        return rhs_G0
    
    def rhs_GL(self, rho: torch.Tensor) -> torch.Tensor:        
        A_op_rho = torch.einsum('ij,klmj->klmi', self.A_op, rho)
        rhs_GL = self.params['R0'] * self.xi_r * A_op_rho
        return rhs_GL
    
    def rhs_GNL(self, rho: torch.Tensor) -> torch.Tensor:
        S_dot_rho = torch.einsum('i,klmi->klm', self.S_z, rho)
        S_dot_rho_expanded = S_dot_rho.unsqueeze(-1)
        A_SE_rho = torch.einsum('ij,klmj->klmi', self.A_SE, rho)
        rhs_GNL = -self.params['eta'] * S_dot_rho_expanded * A_SE_rho
        return rhs_GNL
    
    def __call__(self, rho: torch.Tensor) -> torch.Tensor:
        """计算空间离散化的右端项"""
        rho = rho.to(self.device)
        
        diffusion = self.rhs_diffusion(rho)
        G0_term = self.rhs_G0(rho)
        GL_term = self.rhs_GL(rho)
        GNL_term = self.rhs_GNL(rho)
        
        spatial_rhs = torch.zeros_like(rho, device=self.device)
        interior = slice(1, -1)
        
        spatial_rhs[interior, interior, interior, :] = (
            diffusion 
            - G0_term[interior, interior, interior, :] 
            - GL_term[interior, interior, interior, :]
            - GNL_term[interior, interior, interior, :]
        )
        
        return spatial_rhs
    



def create_spatial_discretization(config: Config, grid_data: Dict) -> Spatial_fun:
    """Factory function to create a SpatialDiscretization instance."""
    return Spatial_fun(
        config=config,
        grid=grid_data['grid'],
        grid_x=grid_data['grid_x'],
        grid_y=grid_data['grid_y'],
        grid_z=grid_data['grid_z'],
        dx=grid_data['dx'],
        dy=grid_data['dy'],
        dz=grid_data['dz'],
        Nx=grid_data['grid'].shape[0],
        Ny=grid_data['grid'].shape[1],
        Nz=grid_data['grid'].shape[2]
    )