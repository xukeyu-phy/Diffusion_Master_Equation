import torch
from config import Config
import time


class DMESolver:
    def __init__(self, device, dtype, phy_ps, data_dir):
        self.device = device
        self.config = Config(device, dtype)
        torch.set_default_dtype(dtype)

        self.config._create_non_uniform_grid(data_dir)
        self.config._setup_coefficient()
        self._init_phy_ps(phy_ps)
        

    def _init_phy_ps(self, phy_ps):
        self.config._setup_phy_ps(phy_ps)
        self.config._setup_matrices(self.config.Qa, self.config.Qb)
        self._update_pump_distribution()


    def _main_line(self, dt):
        start_time = time.time()
        rho_init = torch.zeros((self.config.Nx, self.config.Ny, self.config.Nz, 8), device=self.device)
        rho_init = self._setup_initial_condition(rho_init)
        rho_init = self._setup_boundary_conditions(rho_init, self.config.ghostcell, self.config.bc_type, self.config.bc_value)

        t = 0.0
        t_iter = 0
        rho_n = rho_init.clone()        

        while t < self.config.T_final:
            if (t + dt) >= self.config.T_final:
                dt = self.config.T_final - t
            t += dt
            t_iter += 1       

            if t_iter % 500 == 0:
                            
                rho_n0 = rho_n.clone()               
                rho_n = self.runge_kutta_2_step(rho_n, dt, self.config.ghostcell, self.config.bc_type)
                drho = torch.abs(rho_n - rho_n0)
                inf_drho = torch.max(drho)
                l2_drho = torch.norm(drho, p=2)
                print(f"Iter: {t_iter}, Time: {t:.6f}, l2_drho = {l2_drho:.4e}, inf_drho = {inf_drho:.4e} ")
                
                if l2_drho < self.config.convergence_tol:
                    print(f'Convergence: l2_drho = {l2_drho:.6e}')
                    break
            else:
                rho_n = self.runge_kutta_2_step(rho_n, dt, self.config.ghostcell, self.config.bc_type)
        end_time = time.time()
        runtime =  end_time - start_time
        print(f"Runtime: {runtime:.3f}s")    
        return rho_n
    



    def _update_pump_distribution(self,isapreature = False,isiteration = False):
        if isiteration == False:
            x, y, z = self.config.grid_x, self.config.grid_y, self.config.grid_z
            self.xi_r = torch.exp(-2*(x**2 + y**2)/(self.config.w**2)) * torch.exp(-self.config.OD * z)

        elif isiteration == True:
            return 1 

        if len(self.xi_r.shape) == 3:
            self.xi_r = self.xi_r.unsqueeze(-1)


    def rhs(self, rho):
        rho = rho.to(self.device)
        
        diffusion = self.rhs_diffusion(rho)
        G0_term = self.rhs_G0(rho)
        GL_term = self.rhs_GL(rho)
        GNL_term = self.rhs_GNL(rho)
        
        spatial_rhs = torch.zeros_like(rho, device=self.device)
        interior = slice(1, -1)
        
        spatial_rhs[interior, interior, interior, :] = (
            diffusion[interior, interior, interior, :]
            - G0_term[interior, interior, interior, :] 
            - GL_term[interior, interior, interior, :]
            - GNL_term[interior, interior, interior, :]
        )
        
        return spatial_rhs

    
    def _setup_boundary_conditions(self, rho, gc, bc_type, bc_value):
        if bc_type is None:
            bc_type = self.config.bc_type
        
        if bc_type == 'dirichlet':
            # x direction 
            rho[0, :, :, :] = self.config.bc_value
            rho[-1, :, :, :] = self.config.bc_value

            # y direction
            rho[:, 0, :, :] = self.config.bc_value
            rho[:, -1, :, :] = self.config.bc_value

            # z direction
            rho[:, :, 0, :] = self.config.bc_value
            rho[:, :, -1, :] = self.config.bc_value
            
        elif bc_type == 'periodic':
            rho[:gc, :, :, :] = rho[-2*gc:-gc, :, :, :] 
            rho[-gc:, :, :, :] = rho[gc:2*gc, :, :, :]   
            rho[:, :gc, :, :] = rho[:, -2*gc:-gc, :, :]  
            rho[:, -gc:, :, :] = rho[:, gc:2*gc, :, :]              
            rho[:, :, :gc, :] = rho[:, :, -2*gc:-gc, :]  
            rho[:, :, -gc:, :] = rho[:, :, gc:2*gc, :]   

        elif bc_type == 'neumann':
            pass
        else:
            raise ValueError(f"Unsupported boundary condition type: {bc_type}")
        
        return rho
    
    def _setup_initial_condition(self, rho: torch.Tensor) -> torch.Tensor:
        if self.config.initial_condition_type == 'uniform':
            rho[:, :, :, :] = self.config.initial_value
        elif self.config.initial_condition_type == 'analytical':
            pass        
        return rho
    
    def _setup_steady_state_solution(self) -> torch.Tensor:
        x = 0.0
        y = 0.0
        z = 0.5
        xi_r0 = torch.exp(-(x**2 + y**2)/(2*self.config.w**2)) * torch.exp(-self.config.OD * z)
        xi_r0 = xi_r0.clone().detach().to(dtype=torch.float64, device=self.device)
        R = self.config.R0 * xi_r0
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
    
    def rhs_diffusion(self, rho: torch.Tensor):
        # Calculating second derivative
        d2rho_dx2 = (self.config.alpha_x * rho[:-2, 1:-1, 1:-1, :] + 
                     self.config.beta_x * rho[1:-1, 1:-1, 1:-1, :] + 
                     self.config.gamma_x * rho[2:, 1:-1, 1:-1, :])
        
        d2rho_dy2 = (self.config.alpha_y * rho[1:-1, :-2, 1:-1, :] + 
                     self.config.beta_y * rho[1:-1, 1:-1, 1:-1, :] + 
                     self.config.gamma_y * rho[1:-1, 2:, 1:-1, :])
        
        d2rho_dz2 = (self.config.alpha_z * rho[1:-1, 1:-1, :-2, :] + 
                     self.config.beta_z * rho[1:-1, 1:-1, 1:-1, :] + 
                     self.config.gamma_z * rho[1:-1, 1:-1, 2:, :])
        nabla2 = rho.clone()
        nabla2[1:-1, 1:-1, 1:-1, :] = d2rho_dx2 + d2rho_dy2 + d2rho_dz2
        return self.config.D * nabla2
    
    def rhs_G0(self, rho):
        self.G0_matrix = (1 + self.config.eta) * self.config.A_SD + self.config.fD * self.config.A_FD
        rhs_G0 = torch.einsum('ij,klmj->klmi', self.G0_matrix, rho)
        return rhs_G0
    
    def rhs_GL(self, rho):        
        A_op_rho = torch.einsum('ij,klmj->klmi', self.config.A_op, rho)
        rhs_GL = self.config.R0 * self.xi_r * A_op_rho
        return rhs_GL
    
    def rhs_GNL(self, rho):
        S_dot_rho = torch.einsum('i,klmi->klm', self.config.S_z, rho)
        S_dot_rho_expanded = S_dot_rho.unsqueeze(-1)
        A_SE_rho = torch.einsum('ij,klmj->klmi', self.config.A_SE, rho)
        rhs_GNL = -self.config.eta * S_dot_rho_expanded * A_SE_rho
        return rhs_GNL
    
    def rhs_GNL_deim(self, rho):
        self.config._setup_matrices(1, 1)
        rho = rho.to(self.device)

        S_dot_rho = torch.einsum('i,klmi->klm', self.config.S_z, rho)
        S_dot_rho_expanded = S_dot_rho.unsqueeze(-1)
        A_SE_rho = torch.einsum('ij,klmj->klmi', self.config.A_SE, rho)
        rhs_GNL_deim = S_dot_rho_expanded * A_SE_rho
        return rhs_GNL_deim

    def runge_kutta_2_step(self, rho_n, dt, ghostcell, bc_type):
        """
        2 order Runge-Kutta time integration
        """
        # Stage 1
        k1 = self.rhs(rho_n)
        rho_1 = rho_n + 1.0 * dt * k1
        rho_1 = self._setup_boundary_conditions(rho_1, ghostcell, bc_type)
        
        # Stage 2
        k2 = self.rhs(rho_1)
        
        # Final combination
        rho_new = rho_n + (dt/2.0) * (k1 + k2)
        rho_new = self._setup_boundary_conditions(rho_new, ghostcell, bc_type)
        
        return rho_new

 