import torch
from typing import Dict, Tuple, Optional
# from config import Config, device, dtype
from config import Config
import time
from pathlib import Path
from scipy import sparse


class ROMSolver:
    def __init__(self, r, rf, Phi_r, Phi_f, P_f, phy_ps, device, dtype, pod_dir, result_dir):
        self.config = Config(device, dtype)
        self.r = r
        self.rf = rf
        self.Phi_r = Phi_r.to(device)
        self.Phi_f = Phi_f.to(device)
        self.P_f = P_f




        self.Nh = Phi_r.shape[0]  # 8*N^3
        self.N3 = self.Nh // 8      # N^3
        self.N = round(self.N3 ** (1/3))

        self.config._create_non_uniform_grid(result_dir)
        self.config._setup_coefficient()
        
        self.device = device
        # self.DA = self.L_space()
        # self.precompute_reduced_operators(self.Phi_r)
        # self.D_r = self.rom_diffusion_pre()

        self._init_phy_ps(phy_ps)
        self.snapshots_means = torch.load(pod_dir/'snapshots_means.pt', weights_only=True).to(self.device)
        rho_mean = self.snapshots_means.reshape(self.N, self.N, self.N, 8)
        self.rhs_mean = self.rom_rhs_mean_field(rho_mean)
        self.rom_GNL_preperform_NQE(rho_mean)
        self.nabla_r = self.rom_diffusion_preperform()
        self.G0_r = self.rom_G0_preperform()
        self.GL_r = self.rom_GL_preperform()
        self.GNLTmp = self.rom_GNL_preperform()

    def _init_phy_ps(self, phy_ps):
        self.config._setup_phy_ps(phy_ps)
        self.config._setup_matrices(self.config.Qa, self.config.Qb)
        self._update_pump_distribution()

    def _main_line(self, dt):
        start_time = time.time()
        iter_count = 0
        for iter_count in range(10):
            rho_init = torch.zeros(self.Nh, 1).to(self.device)
            rho_init = self._setup_initial_condition(rho_init)
            rho_r_init = self.Phi_r.T @ (rho_init - self.snapshots_means.reshape(-1,1)) 

            t = 0.0
            t_iter = 0
            rho_r_n = rho_r_init.clone()    
            relative_drho_0 = 1
            while t < self.config.T_final:
                if (t + dt) >= self.config.T_final:
                    dt = self.config.T_final - t
                t += dt
                t_iter += 1 

                if t_iter % 500 == 1:
                                
                    rho_r_n0 = rho_r_n.clone()               
                    rho_r_n = self.runge_kutta_2_step(rho_r_n, dt, self.config.ghostcell, self.config.bc_type)
                    drho = torch.abs(rho_r_n - rho_r_n0)
                    inf_drho = torch.max(drho)
                    l2_drho = torch.norm(drho, p=2)
                    # inf_relative_error = inf_drho / torch.max(rho_r_n0)
                    # l2_relative_drho = l2_drho / torch.norm(rho_r_n, p=2)
                    # print(f"Iter: {t_iter}, Time: {t:.6f}, l2_drho = {l2_drho:.4e}, inf_drho = {inf_drho:.4e} , inf_relative_error = {inf_relative_error * 100 :.5f} %")
                    print(f"Iter: {t_iter}, Time: {t:.6f}, l2_drho = {l2_drho:.4e}, inf_drho = {inf_drho:.4e}")
                    
                    
                    # if l2_drho > relative_drho_0:
                    if l2_drho < self.config.convergence_tol:
                        print(f'Convergence: l2_drho = {l2_drho:.6e}')
                        break
                    if t_iter >= 100000:
                        break
                    # l2_relative_drho = l2_relative_drho
                else:
                    rho_r_n = self.runge_kutta_2_step(rho_r_n, dt, self.config.ghostcell, self.config.bc_type)
            iter_count += 1        
        end_time = time.time()
        runtime =  end_time - start_time
        print(f"Runtime: {runtime:.3f}s, 1 case time: {runtime/iter_count:3f}")

        return rho_r_n

    def _setup_initial_condition(self, rho: torch.Tensor) -> torch.Tensor:
        if self.config.initial_condition_type == 'uniform':
            rho[:, :] = self.config.initial_value
        elif self.config.initial_condition_type == 'analytical':
            pass        
        return rho

    def _update_pump_distribution(self,isapreature = False,isiteration = False):
        if isiteration == False:
            x, y, z = self.config.grid_x, self.config.grid_y, self.config.grid_z
            xi = torch.exp(-2*(x**2 + y**2)/(self.config.w**2)) * torch.exp(-self.config.OD * z)

        elif isiteration == True:
            return 1 #后续在此函数当中更新扩散方程的迭代

        if len(xi.shape) == 3:
            xi = xi.unsqueeze(-1)
        # if isapreature:s
        #     apreature = x**2 + y**2 <= self.config.w_a**2
        #     self.xi_r = self.xi_r*apreature
        xi = torch.tile(xi, (1, 1, 1, 8))
        self.xi = xi.reshape(-1, 1)  # 展平为向量
        # return self.xi_r

    def rom_diffusion_preperform(self, bc_value=0.125):     
        Phi_space = self.Phi_r.reshape(self.N, self.N, self.N, 8, self.r)
        APhi = Phi_space.clone()
        self.config.alpha_x = self.config.alpha_x.unsqueeze(-2) 
        self.config.beta_x = self.config.beta_x.unsqueeze(-2)
        self.config.gamma_x = self.config.gamma_x.unsqueeze(-2)
        self.config.alpha_y = self.config.alpha_y.unsqueeze(-2) 
        self.config.beta_y = self.config.beta_y.unsqueeze(-2)
        self.config.gamma_y = self.config.gamma_y.unsqueeze(-2)
        self.config.alpha_z = self.config.alpha_z.unsqueeze(-2) 
        self.config.beta_z = self.config.beta_z.unsqueeze(-2)
        self.config.gamma_z = self.config.gamma_z.unsqueeze(-2)
        APhi_dx = (  self.config.alpha_x * Phi_space[:-2, 1:-1, 1:-1, :, :] + 
                     self.config.beta_x * Phi_space[1:-1, 1:-1, 1:-1, :, :] + 
                     self.config.gamma_x * Phi_space[2:, 1:-1, 1:-1, :, :])
        
        APhi_dy = (  self.config.alpha_y * Phi_space[1:-1, :-2, 1:-1, :, :] + 
                     self.config.beta_y * Phi_space[1:-1, 1:-1, 1:-1, :, :] + 
                     self.config.gamma_y * Phi_space[1:-1, 2:, 1:-1, :, :])
        
        APhi_dz = (  self.config.alpha_z * Phi_space[1:-1, 1:-1, :-2, :, :] + 
                     self.config.beta_z * Phi_space[1:-1, 1:-1, 1:-1, :, :] + 
                     self.config.gamma_z * Phi_space[1:-1, 1:-1, 2:, :, :])
        APhi[1:-1, 1:-1, 1:-1, :, :] = (APhi_dx + APhi_dy + APhi_dz)    
        APhi = APhi.reshape(-1, self.r)
        nabla_r = self.config.D * self.Phi_r.T @ APhi
        return nabla_r

    def rom_diffusion_mean_preperform(self, rho_mean, bc_value=0.125):     
        Arho_mean = rho_mean.clone()
        Arho_mean_dx = (  self.config.alpha_x * rho_mean[:-2, 1:-1, 1:-1, :] + 
                     self.config.beta_x * rho_mean[1:-1, 1:-1, 1:-1, :] + 
                     self.config.gamma_x * rho_mean[2:, 1:-1, 1:-1, :])
        
        Arho_mean_dy = (  self.config.alpha_y * rho_mean[1:-1, :-2, 1:-1, :] + 
                     self.config.beta_y * rho_mean[1:-1, 1:-1, 1:-1, :] + 
                     self.config.gamma_y * rho_mean[1:-1, 2:, 1:-1, :])
        
        Arho_mean_dz = (  self.config.alpha_z * rho_mean[1:-1, 1:-1, :-2, :] + 
                     self.config.beta_z * rho_mean[1:-1, 1:-1, 1:-1, :] + 
                     self.config.gamma_z * rho_mean[1:-1, 1:-1, 2:, :])
        Arho_mean[1:-1, 1:-1, 1:-1, :] = (Arho_mean_dx + Arho_mean_dy + Arho_mean_dz)    
        Arho_mean = Arho_mean.reshape(-1, 1)
        diff_mean = self.config.D * self.Phi_r.T @ Arho_mean
        return diff_mean

    def rom_G0_preperform(self): 
        G0_matrix = (1 + self.config.eta) * self.config.A_SD + self.config.fD * self.config.A_FD  
        Phi_r_reshaped = self.Phi_r.reshape(self.N, self.N, self.N, 8, self.r)  
        
        Tmp = torch.einsum('id,abcde->abcie', G0_matrix, Phi_r_reshaped)
        Tmp_matrix = Tmp.reshape(self.Nh, self.r)  # shape: (8*N^3, r)        
        self.G0_r = torch.matmul(self.Phi_r.T, Tmp_matrix)  # shape: (r, r)
        return self.G0_r

    def rom_G0_mean_preperform(self, rho_mean): 
        G0_matrix = (1 + self.config.eta) * self.config.A_SD + self.config.fD * self.config.A_FD  
        Tmp = torch.einsum('id,abcd->abci', G0_matrix, rho_mean)
        Tmp_matrix = Tmp.reshape(self.Nh, 1)          
        G0_mean = torch.matmul(self.Phi_r.T, Tmp_matrix)  # shape: (r, r)
        return G0_mean
    
    def rom_GL_preperform(self):
        Phi_r_reshaped = self.Phi_r.reshape(self.N, self.N, self.N, 8, self.r)  
        xi = self.xi        
        A_op_Phi = torch.einsum('id,abcde->abcie', self.config.A_op, Phi_r_reshaped)
        A_op_Phi = A_op_Phi.reshape(-1, self.r)
        xi_APhi = xi * A_op_Phi
        self.GL_r = self.config.R0 * self.Phi_r.T @ xi_APhi
        return self.GL_r

    def rom_GL_mean_preperform(self, rho_mean):        
        xi = self.xi
        A_op_mean = torch.einsum('id,abcd->abci', self.config.A_op, rho_mean)
        A_op_mean = A_op_mean.reshape(-1, 1)
        xi_A_mean = xi * A_op_mean
        GL_mean = self.config.R0 * self.Phi_r.T @ xi_A_mean
        return GL_mean
    
    def rom_GNL_preperform(self):
        self.GNLTmp = self.Phi_r.T @ self.Phi_f @ torch.linalg.inv(self.Phi_f[self.P_f, :])
        num_components = 8        

        spatial_indices = torch.div(self.P_f, num_components, rounding_mode='floor')
        self.comp_indices = self.P_f % num_components
        

        unique_spatial_indices, self.inverse_indices = torch.unique(spatial_indices, return_inverse=True)

        self.spatial_rows = []
        for idx in unique_spatial_indices:
            self.spatial_rows.extend(range(idx * num_components, (idx + 1) * num_components))

        return self.GNLTmp


    def rhs(self, rho):
        # rhs_vib =  - self.rom_rhs_G0(rho) - self.rom_rhs_GL(rho) - self.rom_rhs_GNL(rho) + self.rom_rhs_diffusion(rho)
        rhs_vib =  - self.rom_rhs_G0(rho) - self.rom_rhs_GL(rho) - self.rom_rhs_GNL_NQE(rho) + self.rom_rhs_diffusion(rho) 
        # rhs = - self.rom_rhs_G0(rho) - self.rom_rhs_GL(rho) - self.rom_rhs_GNL(rho) 
        rhs = rhs_vib + self.rhs_mean
        # rhs = rhs_vib
        return rhs
    


    def rom_rhs_diffusion(self, rho_r):        
        rho_r = self.nabla_r @ rho_r
        return rho_r

    def rom_rhs_G0(self, rho_r):
        rhs_G0_r = self.G0_r @ rho_r        
        return rhs_G0_r  

    def rom_rhs_GL(self, rho_r):
        rhs_GL_r = self.GL_r @ rho_r
        return rhs_GL_r

    def rom_rhs_GNL_ori(self, rho_r):
        rho_h = self.Phi_r @ rho_r + self.snapshots_means.reshape(-1,1)
        rho_h = rho_h.reshape(self.N, self.N, self.N, 8)
        S_dot_rho = torch.einsum('i,klmi->klm', self.config.S_z, rho_h)
        A_SE_rho = torch.einsum('ij,klmj->klmi', self.config.A_SE, rho_h)

        S_dot_rho_flat = S_dot_rho.unsqueeze(-1)
        A_SE_rho_flat = A_SE_rho.reshape(-1)          

        S_dot_rho_expanded = torch.tile(S_dot_rho_flat, (1, 1, 1, 8))       
        S_dot_rho_expanded = S_dot_rho_expanded.reshape(-1)  # shape: (8*N3,)
        F_full = S_dot_rho_expanded * A_SE_rho_flat  # shape: (8*N3,)
        
        F_P = F_full[self.P_f]  # shape: (f,)
        F_P = F_P.reshape(-1, 1)
        rhs_GNL_r = -self.config.eta * self.GNLTmp @ F_P  # shape: (r,)

        return rhs_GNL_r
    
    def rom_rhs_GNL(self, rho_r):
        rho_h_spatial = self.Phi_r[self.spatial_rows] @ rho_r + self.snapshots_means[self.spatial_rows].reshape(-1, 1)
        rho_h_spatial = rho_h_spatial.reshape(-1, 8) 
        
        S_dot_rho_spatial = torch.einsum('i,ji->j', self.config.S_z, rho_h_spatial)  
        A_SE_rho_spatial = torch.einsum('ij,kj->ki', self.config.A_SE, rho_h_spatial)  

        S_dot_rho_P = S_dot_rho_spatial[self.inverse_indices] 
        A_SE_rho_P = A_SE_rho_spatial[self.inverse_indices, self.comp_indices] 
        F_P = S_dot_rho_P * A_SE_rho_P 
        F_P = F_P.reshape(-1, 1)

        rhs_GNL_r = -self.config.eta * self.GNLTmp @ F_P  
    
        return rhs_GNL_r

    def rom_rhs_GNL_NQE(self, rho_r):
   
        # A_NQE_rho_r = torch.zeros(self.r, 1).to(self.device)
        # for i in range(self.r):
        #     for j in range(self.r):
        #         idx = i * self.r + j
        #         A_NQE_rho_r += self.A_NQE[:, idx:idx+1] * rho_r[i] * rho_r[j]
        rho_r_flat = rho_r.flatten()
        rho_outer = torch.outer(rho_r_flat, rho_r_flat)
        rho_outer_flat = rho_outer.flatten().unsqueeze(0) 
        A_NQE_rho_r = self.A_NQE @ rho_outer_flat.T

        B_NQE_rho_r = self.B_NQE @ rho_r

        rhs_GNL_r =  -self.config.eta * (A_NQE_rho_r + B_NQE_rho_r + self.C_NQE_rho_r)

        return rhs_GNL_r
    
    def rom_GNL_preperform_NQE(self, rho_mean):
        Phi_r_reshaped = self.Phi_r.reshape(self.N3, 8, self.r)  
        S_dot_Phi_r = torch.einsum('b,abc->ac', self.config.S_z, Phi_r_reshaped)  
        S_dot_Phi_r = S_dot_Phi_r.unsqueeze(1)
        S_dot_Phi_r = S_dot_Phi_r.expand(-1, 8, -1)
        A_SE_Phi_r = torch.einsum('db,abc->adc', self.config.A_SE, Phi_r_reshaped)  
        S_dot_Phi_r = S_dot_Phi_r.reshape(-1, self.r) 
        A_SE_Phi_r = A_SE_Phi_r.reshape(-1, self.r)

        A_NQE_Phi_r = torch.zeros(self.Nh, self.r * self.r).to(self.device)
        for i in range(self.r):
            for j in range(self.r):
                A_NQE_Phi_r[:, i * self.r + j] = S_dot_Phi_r[:, i] * A_SE_Phi_r[:, j]
        self.A_NQE = self.Phi_r.T @ A_NQE_Phi_r   

        A_SE_rho = torch.einsum('id,abcd->abci', self.config.A_SE, rho_mean)
        A_SE_rho = A_SE_rho.reshape(-1, 1)
        B_NQE_1 = S_dot_Phi_r * A_SE_rho

        S_dot_rho = torch.einsum('d,abcd->abc', self.config.S_z, rho_mean)
        S_dot_rho = S_dot_rho.unsqueeze(-1)
        S_dot_rho = S_dot_rho.expand(-1, -1, -1, 8)
        S_dot_rho = S_dot_rho.reshape(-1, 1)
        B_NQE_2 = S_dot_rho * A_SE_Phi_r
        self.B_NQE = self.Phi_r.T @ (B_NQE_1 + B_NQE_2)

        C_NQE = S_dot_rho * A_SE_rho
        C_NQE = self.Phi_r.T @ C_NQE
        self.C_NQE_rho_r = C_NQE

    
    def rom_rhs_mean_field(self, rho_mean):
        diff_mean = self.rom_diffusion_mean_preperform(rho_mean)
        G0_mean = self.rom_G0_mean_preperform(rho_mean)
        GL_mean = self.rom_GL_mean_preperform(rho_mean)
        mean = - G0_mean - GL_mean + diff_mean
        return mean    



    def runge_kutta_2_step(self, rho_n, dt, ghostcell, bc_type):
        """
        2 order Runge-Kutta time integration
        """
        # Stage 1
        k1 = self.rhs(rho_n)
        rho_1 = rho_n + 1.0 * dt * k1
        # rho_1 = self._setup_boundary_conditions(rho_1, ghostcell, bc_type)
        
        # Stage 2
        k2 = self.rhs(rho_1)
        
        # Final combination
        rho_new = rho_n + (dt/2.0) * (k1 + k2)
        # rho_new = self._setup_boundary_conditions(rho_new, ghostcell, bc_type)
        
        return rho_new
    
