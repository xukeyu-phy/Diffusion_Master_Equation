import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import matplotlib.pyplot as plt
from ROM_solver import ROMSolver
from config import Config
from DME_solver import DMESolver

current_dir = Path(__file__).parent
pod_dir = current_dir / 'pod_result'
result_dir = current_dir / "result-deim"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

def compute_rom_for_r(r, rf, Phi_h, Phi_F, P_F, phy_dict, snapshots_means, config, device, dtype, pod_dir, result_dir):
    Phi_r = Phi_h[:, :r]
    Phi_f = Phi_F[:, :rf]
    P_f = P_F[:rf]
    
    romsolver = ROMSolver(r, rf, Phi_r, Phi_f, P_f, phy_dict, device, dtype, pod_dir, result_dir)
    
    dt = config._get_time_step(phy_dict['D'])
    result = romsolver._main_line(dt)
    
    Nh = Phi_r.shape[0]
    N3 = Nh // 8
    N = round(N3 ** (1/3))
    
    recon_rho = Phi_r @ result.cpu() + snapshots_means
    recon_rho = recon_rho.reshape(N, N, N, 8)    
    return recon_rho

def calculate_errors(rom_result, hf_result):
    abs_error = torch.abs(rom_result - hf_result)
    max_error = torch.max(abs_error).item()
    average_error = torch.mean(abs_error).item()    
    return max_error, average_error

def main():
    Phi_h = torch.load(pod_dir / "Phi_h.pt", weights_only=True)
    Phi_F = torch.load(pod_dir / "Phi_F.pt", weights_only=True)
    P_F = torch.load(pod_dir / "P_F.pt", weights_only=True)
    
    param_filename = current_dir / 'parameters.json'
    with open(param_filename, 'r') as f:
        phy_dict = json.load(f)
    
    snapshots_means = torch.load(pod_dir / "snapshots_means.pt", weights_only=True)
    hf_rho = torch.load(result_dir/'DME_HFresult.pt', weights_only=True, map_location=device)
    
    config = Config(device, dtype)
    config._create_non_uniform_grid(result_dir)
    
    rf = 15

    max_errors = []
    average_errors = []
    r_values = list(range(1, 11))
    

    for r in r_values:
        print(f"计算 r = {r} 的ROM解...")
        
        try:
            recon_rho = compute_rom_for_r(r, rf, Phi_h, Phi_F, P_F, phy_dict, 
                                         snapshots_means, config, device, dtype, 
                                         pod_dir, result_dir)
            

            max_error, average_error = calculate_errors(recon_rho, hf_rho)            
            max_errors.append(max_error)
            average_errors.append(average_error)
            
            print(f"r = {r}: 最大误差 = {max_error:.6e}, 平均误差 = {average_error:.6e}")
            
        except Exception as e:
            print(f"计算 r = {r} 时出错: {e}")
            max_errors.append(np.nan)
            average_errors.append(np.nan)
    

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, max_errors, 'o-', label='Max Error')
    plt.plot(r_values, average_errors, 's-', label='Average Error')
    plt.xticks(r_values)
    plt.xlabel('Reduced Space Dimension (r)')
    plt.ylabel('Error')
    plt.title('Basis Refinement Error Analysis with DEIM')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  

    plt.savefig(result_dir / 'error_analysis_deim.png', dpi=300, bbox_inches='tight')
    plt.show()
    

    error_data = {
        'r_values': r_values,
        'max_errors': max_errors,
        'average_errors': average_errors
    }
    torch.save(error_data, result_dir / 'error_analysis_data.pt')
    
    print("基函数加密测试完成!")

if __name__ == "__main__":
    main()