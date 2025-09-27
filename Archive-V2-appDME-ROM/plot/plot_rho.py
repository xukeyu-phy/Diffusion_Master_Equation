import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple, Optional
# from config import device, dtype 

current_dir = Path(__file__).parent
device = torch.device('cpu')
dtype = torch.float64
torch.set_default_dtype(dtype)
pod_dir = current_dir / 'svd_results'
results_dir = current_dir / "result"


def initialize_system(timestamp: str) -> Tuple[torch.device, Path, Path, torch.Tensor, Dict, torch.Tensor]:
    """
    Initialize the entire system including:
    1. Setting computation device and default data type
    2. Preparing data and figure directories
    3. Loading data files
    
    Args:
        timestamp: Timestamp string for creating figure directory
    
    Returns:
        device: Computation device (GPU/CPU)
        data_dir: Path to data directory
        fig_dir: Path to figures directory
        rho: Loaded density data
        Mesh_data: Mesh data dictionary
        P: Physical quantity P data
    """
    
    # data_dir = current_dir / "data"
    # data_dir = current_dir 
    fig_dir = current_dir / f"fig_{timestamp}"  
    fig_dir.mkdir(exist_ok=True)

    rho = torch.load(results_dir/'DME_ROM_projection.pt', weights_only=True, map_location=device)
    Mesh_data = torch.load(results_dir/'Mesh.pt', weights_only=True, map_location=device)
    # P = torch.load(data_dir/'P.pt', weights_only=True)
    
    # return device, data_dir, fig_dir, rho, Mesh_data, P
    return device, fig_dir, rho, Mesh_data

def find_closest_index(values: torch.Tensor, target: float) -> Tuple[int, float]:
    """Find the index of the value closest to the target"""
    idx = torch.argmin(torch.abs(values - target)).item()
    return idx, values[idx].item()


def plot_contour(data_slice: np.ndarray, coord1: np.ndarray, coord2: np.ndarray, 
                value: float, fig_dir: Path, 
                quantity_name: str = '\\rho_{1}', plane: str = 'YOZ', 
                fixed_coord: str = 'x') -> Path:
    """
    General contour plot function
    
    Parameters:
        data_slice: Slice data to plot
        coord1: First coordinate axis data
        coord2: Second coordinate axis data
        value: Value of the fixed coordinate
        fig_dir: Directory to save figures
        quantity_name: Name of the physical quantity
        plane: Plane name (e.g., 'YOZ', 'XOZ', 'XOY')
        fixed_coord: Name of the fixed coordinate (e.g., 'x', 'y', 'z')
    """
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 8))

    contour = ax.contourf(
        coord1, coord2, data_slice.T, 
        levels=100,
        cmap='viridis',
        antialiased=True,
        extend='both'
    )
    ax.contour(
        coord1, coord2, data_slice.T,
        levels=10,
        colors='white',
        linewidths=0.5,
        alpha=0.5
    )
    
    cbar = fig.colorbar(contour, ax=ax, pad=0.02)
    cbar.set_label(f'{quantity_name} Value', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=12)
    
    # Set axis labels
    if plane == 'YOZ':
        ax.set_xlabel('y', fontsize=14, labelpad=10)
        ax.set_ylabel('z', fontsize=14, labelpad=10)
    elif plane == 'XOZ':
        ax.set_xlabel('x', fontsize=14, labelpad=10)
        ax.set_ylabel('z', fontsize=14, labelpad=10)
    elif plane == 'XOY':
        ax.set_xlabel('x', fontsize=14, labelpad=10)
        ax.set_ylabel('y', fontsize=14, labelpad=10)
    
    # Set title
    ax.set_title(f'The distribution of {quantity_name} in {plane} plane when {fixed_coord} = {value:.2f}', 
                 fontsize=16, pad=20)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    
    save_path = fig_dir / f'{quantity_name}_{plane}.png'  
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return save_path


def plot_density_components(rho: torch.Tensor, grid: torch.Tensor, 
                           coord_values: Tuple[float, float, float],
                           fig_dir: Path) -> Dict[str, Path]:
    """
    Plot the 8 components of rho along three orthogonal directions
    
    Parameters:
        rho: Density tensor (nx, ny, nz, 8)
        grid: Grid coordinate tensor (nx, ny, nz, 3)
        coord_values: Coordinate values (x_val, y_val, z_val)
        fig_dir: Directory to save figures
        
    Returns:
        Dictionary of saved figure paths
    """
    x_val, y_val, z_val = coord_values
    
    grid_np = grid.cpu().numpy()
    x_coords = grid_np[:, 0, 0, 0]
    y_coords = grid_np[0, :, 0, 1]
    z_coords = grid_np[0, 0, :, 2]
    
    x_idx = np.argmin(np.abs(x_coords - x_val))
    y_idx = np.argmin(np.abs(y_coords - y_val))
    z_idx = np.argmin(np.abs(z_coords - z_val))
    
    saved_paths = {}
    
    colors = plt.cm.tab10.colors
    
    # 1. Plot along X direction (fixed Y,Z)
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(8):
        ax.plot(x_coords, rho[:, y_idx, z_idx, i].cpu().numpy(),
                label=f'$\\rho_{i+1}$',
                color=colors[i],
                linestyle='-',
                linewidth=1.2)
    
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('$\\rho$ Value', fontsize=14)
    ax.set_title(f'$\\rho$ Components along X-axis\n(y={y_coords[y_idx]:.2f}, z={z_coords[z_idx]:.2f})', 
                 fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    path = fig_dir / 'rho_components_x.png'  
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    saved_paths['x_components'] = path
    
    # 2. Plot along Y direction (fixed X,Z)
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(8):
        ax.plot(y_coords, rho[x_idx, :, z_idx, i].cpu().numpy(),
                label=f'$\\rho_{i+1}$',
                color=colors[i],
                linestyle='-',
                linewidth=1.2)
    
    ax.set_xlabel('y', fontsize=14)
    ax.set_ylabel('$\\rho$ Value', fontsize=14)
    ax.set_title(f'$\\rho$ Components along Y-axis\n(x={x_coords[x_idx]:.2f}, z={z_coords[z_idx]:.2f})', 
                 fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    path = fig_dir / 'rho_components_y.png'  
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    saved_paths['y_components'] = path
    
    # 3. Plot along Z direction (fixed X,Y)
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(8):
        ax.plot(z_coords, rho[x_idx, y_idx, :, i].cpu().numpy(),
                label=f'$\\rho_{i+1}$',
                color=colors[i],
                linestyle='-',
                linewidth=1.2)
    
    ax.set_xlabel('z', fontsize=14)
    ax.set_ylabel('$\\rho$ Value', fontsize=14)
    ax.set_title(f'$\\rho$ Components along Z-axis\n(x={x_coords[x_idx]:.2f}, y={y_coords[y_idx]:.2f})', 
                 fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    path = fig_dir / 'rho_components_z.png'  
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    saved_paths['z_components'] = path
    
    return saved_paths


def visualize_P_slices(P: torch.Tensor, grid: torch.Tensor, 
                      coord_values: Tuple[float, float, float], 
                      fig_dir: Path) -> Dict[str, Path]:
    """
    Visualize slices of P (without plotting axis components)
    
    Parameters:
        P: Physical quantity P data (Nx, Ny, Nz)
        grid: Grid coordinates
        coord_values: Coordinate values (x_val, y_val, z_val)
        fig_dir: Directory to save figures
        
    Returns:
        Dictionary of saved figure paths
    """
    x_val, y_val, z_val = coord_values
    
    grid_np = grid.cpu().numpy()
    x_coords = grid_np[:, 0, 0, 0]
    y_coords = grid_np[0, :, 0, 1]
    z_coords = grid_np[0, 0, :, 2]
    
    x_idx = np.argmin(np.abs(x_coords - x_val))
    y_idx = np.argmin(np.abs(y_coords - y_val))
    z_idx = np.argmin(np.abs(z_coords - z_val))
    
    saved_paths = {}
    
    # 1. Plot YOZ plane slice (fixed X)
    P_slice = P[x_idx, :, :].cpu().numpy()
    path = plot_contour(P_slice, y_coords, z_coords, x_coords[x_idx], 
                       fig_dir, 'P', 'YOZ', 'x')
    saved_paths['P_yoz_slice'] = path
    
    # 2. Plot XOZ plane slice (fixed Y)
    P_slice = P[:, y_idx, :].cpu().numpy()
    path = plot_contour(P_slice, x_coords, z_coords, y_coords[y_idx], 
                       fig_dir, 'P', 'XOZ', 'y')
    saved_paths['P_xoz_slice'] = path
    
    # 3. Plot XOY plane slice (fixed Z)
    P_slice = P[:, :, z_idx].cpu().numpy()
    path = plot_contour(P_slice, x_coords, y_coords, z_coords[z_idx], 
                       fig_dir, 'P', 'XOY', 'z')
    saved_paths['P_xoy_slice'] = path
    
    return saved_paths


def main():
    
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    device, fig_dir, rho, Mesh_data = initialize_system(timestamp)  
    grid = Mesh_data['grid'].to(device)
    grid_x, grid_y, grid_z = (Mesh_data['grid_x'].to(device), 
                             Mesh_data['grid_y'].to(device), 
                             Mesh_data['grid_z'].to(device))
    
    # Set slice positions
    slice_positions = (0.0, 0.0, 0.5)  # x, y, z
    
    # ------------- Visualize P slices
    # p_paths = visualize_P_slices(P, grid, slice_positions, fig_dir)
    
    # ------------- Visualize all rho plots
    # 1. Plot rho contour maps
        # YOZ plane
    x_idx, x_value = find_closest_index(grid_x[:, 0, 0], slice_positions[0])
    rho_slice = rho[x_idx, :, :, 0].cpu().numpy()
    y_values = grid_y[0, :, 0].cpu().numpy()
    z_values = grid_z[0, 0, :].cpu().numpy()
    rho_path1 = plot_contour(rho_slice, y_values, z_values, x_value, 
                            fig_dir, 'rho_1', 'YOZ', 'x')
    
        # XOZ plane
    y_idx, y_value = find_closest_index(grid_y[0, :, 0], slice_positions[1])
    rho_slice = rho[:, y_idx, :, 0].cpu().numpy()
    x_values = grid_x[:, 0, 0].cpu().numpy()
    rho_path2 = plot_contour(rho_slice, x_values, z_values, y_value, 
                            fig_dir, 'rho_1', 'XOZ', 'y')
    
        # XOY plane
    z_idx, z_value = find_closest_index(grid_z[0, 0, :], slice_positions[2])
    rho_slice = rho[:, :, z_idx, 0].cpu().numpy()
    rho_path3 = plot_contour(rho_slice, x_values, y_values, z_value, 
                            fig_dir, 'rho_1', 'XOY', 'z')
    
    # 2. Plot all 8 components of rho
    rho_components_paths = plot_density_components(rho, grid, slice_positions, fig_dir)
    
    print(f"\nPlots generated successfully:")
    print(f"P slices saved to:")
    # for name, path in p_paths.items():
    #     print(f"  {name}: {path}")
    
    print(f"\nrho plots saved to:")
    print(f"  rho YOZ slice: {rho_path1}")
    print(f"  rho XOZ slice: {rho_path2}")
    print(f"  rho XOY slice: {rho_path3}")
    for name, path in rho_components_paths.items():
        print(f"  {name}: {path}")
    
    print(f"\nTotal execution time: {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    main()