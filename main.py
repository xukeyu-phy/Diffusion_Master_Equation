import torch
import numpy as np
from pathlib import Path
import time
import logging

from config import Config, device, dtype
from spatial_discrete import create_spatial_discretization

torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True       
torch.set_float32_matmul_precision('high')

current_dir = Path(__file__).parent
data_dir = current_dir / "data"
data_dir.mkdir(exist_ok=True)


def exact_solution(t, x, y, z, n, m, l, nu):
    """
    Exact solution for the 3D-diffusion equation.
    """
    a = torch.tensor(nu * (n**2 + m**2 + l**2) * torch.pi**2)
    u = torch.exp(-a*t) * torch.sin(n * torch.pi * x) * torch.sin(m * torch.pi * y) * torch.sin(l * torch.pi * z)
    return u

def runge_kutta_2_step(rho_n, dt, ghostcell, bc_type, spatial_fun):
    """
    2 order Runge-Kutta time integration
    """
    # Stage 1
    k1 = spatial_fun(rho_n)
    rho_1 = rho_n + 1.0 * dt * k1
    rho_1 = spatial_fun._setup_boundary_conditions(rho_1, ghostcell, bc_type)
    
    # Stage 2
    k2 = spatial_fun(rho_1)
    
    # Final combination
    rho_new = rho_n + (dt/2.0) * (k1 + k2)
    rho_new = spatial_fun._setup_boundary_conditions(rho_new, ghostcell, bc_type)
    
    return rho_new


def get_extremes_with_coords(tensor, grid, k=2, largest=True):
    """
    Get top-k extremes and their coordinates
    """
    values, indices = torch.topk(tensor.view(-1), k=k, largest=largest)
    results = []
    for i in range(k):
        idx = np.unravel_index(indices[i].item(), tensor.shape)
        coord = (
            grid[idx[0], idx[1], idx[2], 0].item(),  # x
            grid[idx[0], idx[1], idx[2], 1].item(),  # y
            grid[idx[0], idx[1], idx[2], 2].item()   # z
        )
        results.append((values[i].item(), idx, coord))
    return results


def main():
   
    start_time = time.time()

    config = Config()
    print(f"{'=' * 20} configuration parameter {'=' * 20}")
    print(config)
    
    # n1, n2, n3 = config.n1, config.n2, config.n3
    grid, grid_x, grid_y, grid_z, dx, dy, dz = config.create_non_uniform_grid()
    Nx, Ny, Nz = grid.shape[0], grid.shape[1], grid.shape[2]
    
    grid_data = {
        'grid': grid,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'grid_z': grid_z,
        'dx': dx,
        'dy': dy,
        'dz': dz
    }
    spatial_fun = create_spatial_discretization(config, grid_data)
    
    rho_init = torch.zeros((Nx, Ny, Nz, 8), device=device)
    rho_init = spatial_fun._setup_initial_condition(rho_init)
    rho_init = spatial_fun._setup_boundary_conditions(rho_init, config.ghostcell, config.bc_type, config.bc_value)
    
    min_dx = torch.min(dx)  
    min_dy = torch.min(dy)  
    min_dz = torch.min(dz)
    dd_min = min(min_dx, min_dy, min_dz)
    dt = config.get_time_step(dd_min)
    print(f"time step: {dt:.5e}") 

    t = 0.0
    t_iter = 0
    rho_n = rho_init.clone()
    
    print("\nStart time iteration...")
    while t < config.T_final:
        if (t + dt) >= config.T_final:
            dt = config.T_final - t
        t += dt
        t_iter += 1       

        if t_iter % 500 == 0:
                        
            rho_n0 = rho_n.clone()               
            rho_n = runge_kutta_2_step(rho_n, dt, config.ghostcell, config.bc_type, spatial_fun)
            drho = torch.abs(rho_n - rho_n0)
            inf_drho = torch.max(drho)
            print(f"Iter: {t_iter}, Time: {t:.6f}, inf_drho = {inf_drho:.4e} ")
            
            if inf_drho < config.convergence_tol:
                print(f'Convergence: inf_drho = {inf_drho:.6e}')
                break
        else:
            rho_n = runge_kutta_2_step(rho_n, dt, config.ghostcell, config.bc_type, spatial_fun)
    
    S_z = spatial_fun.S_z
    P = 2 * torch.einsum('i, klmi -> klm', S_z, rho_n)  # Calculate the electric polarization rate

    end_time = time.time()
    runtime =  end_time - start_time
    print(f"Runtime: {runtime:.3f}s")
    print(f'Total iterations: {t_iter} \t Final time: {t:.6f}')

    P = P.cpu()
    torch.save(P, data_dir/'P.pt')    
    rho_n = rho_n.cpu()
    torch.save(rho_n, data_dir/'rho_n.pt')
    print(f'{'-' * 20} Data has been saved! {'-' * 20}')
    
    def save_summary_to_txt():
        # Get the directory where the summary should be saved
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_path = data_dir / f"simulation_summary_{timestamp}.txt"
        
        with open(summary_path, 'w') as f:
            # Write header
            f.write("="*50 + "\n")
            f.write("3D Evolution Simulation Summary\n")
            
            f.write(f"timestamp : {timestamp}\n")
            f.write("="*50 + "\n\n")
            
            # Basic simulation parameters
            f.write(f"[Simulation Parameters]\n")
            f.write(f"BC: {config.bc_type}\n")
            f.write(f"Grid dimensions: n1={config.n1}, n2={config.n2}, n3={config.n3}\n")
            f.write(f"Final time: {config.T_final}\n")
            f.write(f"Diffusion coefficient (nu): {config.D:.6e}\n")
            f.write(f"eta: {config.eta:.5f}\n")
            f.write(f"R0: {config.R0:.5f}\n")
            f.write(f"f_D: {config.f_D:.3f}\n")
            f.write(f"OD: {config.OD:.5f}\n")
            f.write(f"Qa: {config.Qa:.5f}\n")
            f.write(f"Qb: {config.Qa:.5f}\n")
            f.write(f"Time step size: {dt:.6f}\n\n")
            
            # Runtime information
            f.write(f"[Runtime Information]\n")
            f.write(f"Total iterations: {t_iter}\n")
            f.write(f"Physical time reached: {t:.6f}\n")
            f.write(f"Total runtime: {runtime:.3f} seconds\n\n")
            
            # Convergence information
            rho_steady_state = spatial_fun._setup_steady_state_solution()
            f.write(f"[Convergence]\n")
            f.write(f"Final inf_drho: {inf_drho}\n")
            f.write(f"Steady state solution: {rho_steady_state}\n\n")
            
            # Key points analysis
            f.write(f"[Key Points Analysis]\n")
            
            # Center point
            f.write(f"Center point (x={grid[int(Nx/2),0,0,0]:.4f}, y={grid[0,int(Ny/2),0,1]:.4f}, z={grid[0,0,int(Nz/2),2]:.4f}):\n")
            f.write(f"  Sum: {rho_n[int(Nx/2), int(Ny/2), int(Nz/2), :].sum():.6f}\n")
            f.write(f"  Components:\n")
            for i, val in enumerate(rho_n[int(Nx/2), int(Ny/2), int(Nz/2), :]):
                f.write(f"    Component {i}: {val:.4e}\n")
            f.write("\n")
            
            # Center+1 point
            f.write(f"Center+1 point (x={grid[int(Nx/2)+1,0,0,0]:.4f}, y={grid[0,int(Ny/2)+1,0,1]:.4f}, z={grid[0,0,int(Nz/2)+1,2]:.4f}):\n")
            f.write(f"  Sum: {rho_n[int(Nx/2)+1, int(Ny/2)+1, int(Nz/2)+1, :].sum():.6f}\n")
            f.write(f"  Components:\n")
            for i, val in enumerate(rho_n[int(Nx/2)+1, int(Ny/2)+1, int(Nz/2)+1, :]):
                f.write(f"    Component {i}: {val:.4e}\n")
            f.write("\n")
            
            # Origin point
            f.write(f"Origin point (x={grid[0,0,0,0]:.4f}, y={grid[0,0,0,1]:.4f}, z={grid[0,0,0,2]:.4f}):\n")
            f.write(f"  Sum: {rho_n[0, 0, 0, :].sum():.6f}\n")
            f.write(f"  Components:\n")
            for i, val in enumerate(rho_n[0, 0, 0, :]):
                f.write(f"    Component {i}: {val:.4e}\n")
            f.write("\n")
            
            # Far corner point
            f.write(f"Far corner point (x={grid[-1, -1, -1, 0]:.4f}, y={grid[-1, -1, -1, 1]:.4f}, z={grid[-1, -1, -1,2]:.4f}):\n")
            f.write(f"  Sum: {rho_n[-1, -1, -1, :].sum():.6f}\n")
            f.write(f"  Components:\n")
            for i, val in enumerate(rho_n[-1, -1, -1, :]):
                f.write(f"    Component {i}: {val:.4e}\n")
            f.write("\n")
            
            # Extreme value analysis
            rho_sum = rho_n.sum(dim=-1, keepdim=True)    
            max_results = get_extremes_with_coords(rho_sum, grid, k=2, largest=True)
            min_results = get_extremes_with_coords(rho_sum, grid, k=2, largest=False)

            f.write("="*50 + "\n")
            f.write("[Extreme Value Analysis]\n")
            f.write(f"Tensor shape: {rho_sum.shape} | Grid range: x[{grid[...,0].min():.2f}-{grid[...,0].max():.2f}], "
                    f"y[{grid[...,1].min():.2f}-{grid[...,1].max():.2f}], z[{grid[...,2].min():.2f}-{grid[...,2].max():.2f}]\n")
            f.write("="*50 + "\n\n")
            
            f.write("[Maxima]\n")
            for i, (val, idx, (x, y, z)) in enumerate(max_results, 1):
                f.write(f"{'Primary' if i==1 else 'Secondary'} max: {val:.6f}\n")
                f.write(f"  Index: {idx} | Coord: x={x:.4f}, y={y:.4f}, z={z:.4f}\n")
            f.write("\n")
            
            f.write("[Minima]\n")
            for i, (val, idx, (x, y, z)) in enumerate(min_results, 1):
                f.write(f"{'Primary' if i==1 else 'Secondary'} min: {val:.6f}\n")
                f.write(f"  Index: {idx} | Coord: x={x:.4f}, y={y:.4f}, z={z:.4f}\n")
            f.write("\n")
            
            # First component max analysis                
            first_comp = rho_n[..., 0]                                          # Get the maximum index of the first component  rho_n[ :, :, :, 0]
            max_val, max_idx = torch.max(first_comp.view(-1), dim=0)
            max_idx_3d = np.unravel_index(max_idx.item(), first_comp.shape)
    
            all_comps = rho_n[max_idx_3d[0], max_idx_3d[1], max_idx_3d[2], :]
            coord = (
                grid[max_idx_3d[0], max_idx_3d[1], max_idx_3d[2], 0].item(),  # x
                grid[max_idx_3d[0], max_idx_3d[1], max_idx_3d[2], 1].item(),  # y
                grid[max_idx_3d[0], max_idx_3d[1], max_idx_3d[2], 2].item()   # z
            )
            f.write("="*50 + "\n")
            f.write("[First Component Max Analysis]\n")
            f.write(f"Max value of first component: {max_val.item():.6f}\n")
            f.write(f"Location index: {max_idx_3d} | Coord: x={coord[0]:.4f}, y={coord[1]:.4f}, z={coord[2]:.4f}\n")
            f.write(f"All components at this location:\n")
            for i, comp in enumerate(all_comps):
                f.write(f"  Component {i}: {comp.item():.4e}\n")
            f.write("="*50 + "\n\n")
            
            f.write("Simulation completed successfully!\n")
            # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # f.write(f"\n\nReport generated on: {current_time}")
            logging.basicConfig(
                filename='simulation.log',
                level=logging.INFO,
                format='%(asctime)s - %(message)s'
            )
            logging.info("Simulation completed.")
            print(f'{'-' * 20} Simulation Summary has been saved! {'-' * 20}')

    save_summary_to_txt()
    
    print("\nStarting plot generation...")
    import subprocess
    try:
        plot_script_path = Path(__file__).parent / "plot.py"
        subprocess.run(["python", str(plot_script_path)], check=True)
        print("Plot generation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running plot.py: {e}")

    print(f'{'='*20} Congratulations!!! This program has been completed!!!!{'='*20}')

if __name__ == "__main__":
    main()