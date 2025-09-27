import torch
from config import Config
import time
import math

class DMESolver:
    def __init__(self, device, dtype, phy_ps, data_dir):
        self.device = device
        self.config = Config(device, dtype)
        torch.set_default_dtype(dtype)

        self.config._create_non_uniform_grid(data_dir)
        self.config._setup_coefficient()
        self._init_phy_ps(phy_ps)
        self.Nx = self.config.grid_z.shape[0]
        self.Ny = self.config.grid_z.shape[1]
        self.Nz = self.config.grid_z.shape[2]
        

    def _init_phy_ps(self, phy_ps):
        self.config._setup_phy_ps(phy_ps)
        self.config._setup_matrices(self.config.Qa, self.config.Qb)
        self._update_pump_distribution(None, '0', None)


    def _main_line(self, dt):
        start_time = time.time()
        rho_init = torch.zeros((self.config.Nx, self.config.Ny, self.config.Nz, 8), device=self.device)
        rho_init = self._setup_initial_condition(rho_init)
        rho_init = self._setup_boundary_conditions(rho_init, self.config.ghostcell, self.config.bc_type, self.config.bc_value)
        # dt = self.get_time_step(dd_min)

        t = 0.0
        t_iter = 0
        rho_n = rho_init.clone()        

        while t < self.config.T_final:
            if (t + dt) >= self.config.T_final:
                dt = self.config.T_final - t
            t += dt
            t_iter += 1       
            
            if t_iter % 200 == 1:   
                xiz_n0 = self.xi[self.Nx//2, self.Ny//2, :, :]                         
                rho_n0 = rho_n.clone()               
                rho_n = self.runge_kutta_2_step(rho_n, dt, self.config.ghostcell, self.config.bc_type)
                drho = torch.abs(rho_n - rho_n0)
                inf_drho = torch.max(drho)
                l2_drho = torch.norm(drho, p=2)
                
                self._update_pump_distribution(rho_n, '1', None)
                xiz_n = self.xi[self.Nx//2, self.Ny//2, :, :]
                dxiz = torch.abs(xiz_n - xiz_n0)
                l2_dxiz = torch.norm(dxiz, p=2)
                print(f"Iter: {t_iter}, Time: {t:.6f}, l2_drho = {l2_drho:.4e}, l2_dxiz = {l2_dxiz:.4e} ")
                if l2_drho < self.config.convergence_tol and l2_dxiz < self.config.convergence_tol :
                    # rho_n0 = rho_n
                    # rho_n = self.runge_kutta_2_step(rho_n, dt, self.config.ghostcell, self.config.bc_type)
                    # self._update_pump_distribution(rho_n, '1', None)
                    # drho = torch.abs(rho_n - rho_n0)
                    # xiz_n = self.xi[self.Nx//2, self.Ny//2, :, :]
                    # dxiz = torch.abs(xiz_n - xiz_n0)
                    # l2_drho = torch.norm(drho, p=2)
                    # if l2_drho < self.config.convergence_tol:
                    print(f'Convergence: l2_drho = {l2_drho:.6e}, l2_dxiz = {l2_dxiz:.4e}')
                    break
            else:
                # self._update_pump_distribution(rho_n, '1', None)
                rho_n = self.runge_kutta_2_step(rho_n, dt, self.config.ghostcell, self.config.bc_type)

            
        end_time = time.time()
        runtime =  end_time - start_time
        print(f"Runtime: {runtime:.3f}s")    
        return rho_n, self.xi


    def _update_pump_distribution(self, rho=None, isiteration = None, isapreature = False):
        x, y, z = self.config.grid_x, self.config.grid_y, self.config.grid_z
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        z = z.unsqueeze(-1)
        
        xi_xy = torch.exp(-2*(x**2 + y**2)/(self.config.w**2))
        xi_z = torch.ones_like(z, device=self.device)

        if isiteration == 'False':            
            xi_z = torch.exp(-self.config.OD * z)
        elif isiteration == '0':
            xi_z = torch.ones_like(z, device=self.device)
        elif isiteration == '1':
            '''BDF2'''
            dz = self.config.dz
            Srho = torch.einsum('i,klmi->klm', self.config.S_z, rho)
            Srho = Srho.unsqueeze(-1)
            xi_z[:, :, 0, :] = 1.0
            # xi_z[:, :, 1, :] = xi_z[:, :, 0, :] / (1 + self.config.OD * dz[:, :, 0] * (1 - 2 * Srho[:, :, 1, :]))
            # alpha = (2 - math.sqrt(2)) / 2
            # xi_z_alpha = xi_z[:, :, 0, :] / (1 + alpha * self.config.OD * dz[:, :, 0] * (1 - 2 * Srho[:, :, 1, :]))
            # f_alpha = alpha * self.config.OD * dz[:, :, 0] * (1 - 2 * Srho[:, :, 1, :]) * xi_z_alpha
            xi_z[:, :, 1, :] = (xi_z[:, :, 0, :] - xi_z_alpha * (1-alpha) * self.config.OD * dz[:, :, 0] * (1 - 2 * Srho[:, :, 1, :]) )/ (1 + alpha * self.config.OD * dz[:, :, 0] * (1 - 2 * Srho[:, :, 1, :]))
            for ii in range(2, self.Nz-1):
                xi_z[:, :, ii, :] = (4 * xi_z[:, :, ii-1, :] - xi_z[:, :, ii-2, :]) / (3 + 2 * self.config.OD * dz[:, :, ii-1] * (1 - 2 * Srho[:, :, ii, :]))        
            # xi_z[:, :, -1, :] = (4 * xi_z[:, :, -2, :] - xi_z[:, :, -3, :]) / (3 + 2 * self.config.OD * dz[:, :, -1] * (1 - 2 * Srho[:, :, -2, :]))        

        elif isiteration == '2':
            '''Implicit RK2 method'''
            dz = self.config.dz
            Srho = torch.einsum('i,klmi->klm', self.config.S_z, rho)
            Srho = Srho.unsqueeze(-1)
            xi_z[:, :, 0, :] = 1.0
            f1 = 0.5 * self.config.OD * dz[:, :, :] * (1 - 2 * Srho[:, :, :-1, :])
            f2 = 0.5 * self.config.OD * dz[:, :, :] * (1 - 2 * Srho[:, :, 1:, :])
            for ii in range(1, self.Nz):
                xi_z[:, :, ii, :] = xi_z[:, :, ii-1, :] * (1 - f1[:, :, ii-1, :]) / (1 + f2[:, :, ii-1, :])

        elif isiteration == '3':
            '''Intergral'''
            z_points = z[0, 0, :, 0]  
            integrals = torch.zeros((self.Nx, self.Ny, self.Nz), device=self.device)
            dz = z_points[1:] - z_points[:-1]
            for l in range(8):
                S_z_l = self.config.S_z[l]
                rho_component = rho[:, :, :, l]  
                integral_component = torch.zeros((self.Nx, self.Ny, self.Nz), device=self.device)
                
                trapezoid_areas = 0.5 * (rho_component[:, :, :-1] + rho_component[:, :, 1:]) 
                trapezoid_areas = trapezoid_areas * dz.unsqueeze(0).unsqueeze(0) 
                
                for k in range(1, self.Nz):
                    integral_component[:, :, k] = integral_component[:, :, k-1] + trapezoid_areas[:, :, k-1]
                
                integrals += S_z_l * integral_component
                G_OP = torch.exp(2 * self.config.OD * integrals) 
            exp_OD_z = torch.exp(-self.config.OD * z_points)              

            exp_OD_z_broadcast = exp_OD_z.unsqueeze(0).unsqueeze(0).expand(self.Nx, self.Ny, self.Nz)
            xi_z_values = exp_OD_z_broadcast * G_OP 

            xi_z = xi_z_values.unsqueeze(-1)

        elif isiteration == '4':
            '''BDIRK2 - Wrong'''
            dz = self.config.dz
            Srho = torch.einsum('i,klmi->klm', self.config.S_z, rho)
            Srho = Srho.unsqueeze(-1)
            xi_z[:, :, 0, :] = 1.0
            # xi_z[:, :, 1, :] = xi_z[:, :, 0, :] / (1 + self.config.OD * dz[:, :, 0] * (1 - 2 * Srho[:, :, 1, :]))
            alpha = (2 - math.sqrt(2)) / 2
            xi_z_alpha = xi_z[:, :, 0, :] / (1 + alpha * self.config.OD * dz[:, :, 0] * (1 - 2 * Srho[:, :, 1, :]))
            # f_alpha = alpha * self.config.OD * dz[:, :, 0] * (1 - 2 * Srho[:, :, 1, :]) * xi_z_alpha
            xi_z[:, :, 1, :] = (xi_z[:, :, 0, :] - xi_z_alpha * (1-alpha) * self.config.OD * dz[:, :, 0] * (1 - 2 * Srho[:, :, 1, :]) )/ (1 + alpha * self.config.OD * dz[:, :, 0] * (1 - 2 * Srho[:, :, 1, :]))
            for ii in range(2, self.Nz):
                xi_z[:, :, ii, :] = (4 * xi_z[:, :, ii-1, :] - xi_z[:, :, ii-2, :]) / (3 + 2 * self.config.OD * dz[:, :, ii-1] * (1 - 2 * Srho[:, :, ii, :]))        
            xi_z[:, :, -1, :] = (4 * xi_z[:, :, -2, :] - xi_z[:, :, -3, :]) / (3 + 2 * self.config.OD * dz[:, :, -1] * (1 - 2 * Srho[:, :, -2, :])) 
        else:
            print(f'Wrong in Xi_r update!')

        self.xi = xi_xy * xi_z
        # if isapreature:
        #     apreature = x**2 + y**2 <= self.config.w_a**2
        #     self.xi_r = self.xi_r*apreature


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
            rho[:, :, :, :] = self.config.initial_value
        elif self.config.initial_condition_type == 'analytical':
            pass        
        return rho
    
    def _setup_steady_state_solution(self):
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
    
    def rhs_diffusion(self, rho):
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
        rhs_GL = self.config.R0 * self.xi * A_op_rho
        return rhs_GL
    
    def rhs_GNL(self, rho):
        S_dot_rho = torch.einsum('i,klmi->klm', self.config.S_z, rho)
        S_dot_rho_expanded = S_dot_rho.unsqueeze(-1)
        A_SE_rho = torch.einsum('ij,klmj->klmi', self.config.A_SE, rho)
        rhs_GNL = -self.config.eta * S_dot_rho_expanded * A_SE_rho
        return rhs_GNL
    
    def rhs_GNL_deim(self, rho):
        self.config._setup_matrices(self.config.Qa, self.config.Qb)
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

 