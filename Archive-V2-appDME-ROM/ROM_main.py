import os
import json
import torch
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
from ROM_solver import ROMSolver
from config import Config


current_dir = Path(__file__).parent
pod_dir = current_dir / 'pod_result'
result_dir = current_dir / "result"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)




def main():
    # pod_dir = current_dir / 'svd_results'
    Phi_h = torch.load(pod_dir / "Phi_h.pt", weights_only=True)
    Phi_F = torch.load(pod_dir / "Phi_F.pt", weights_only=True)
    P_F = torch.load(pod_dir / "P_F.pt", weights_only=True)

    r = 5
    rf = 5

    Phi_r = Phi_h[:, :r]
    Phi_f = Phi_F[:, :rf]
    P_f = P_F[:rf]

    param_filename = current_dir / 'parameters.json'
    with open(param_filename, 'r') as f:
        phy_dict = json.load(f)
    romsolver = ROMSolver(r, rf, Phi_r, Phi_f, P_f, phy_dict, device, dtype, pod_dir, result_dir)
    config = Config(device, dtype)
    config._create_non_uniform_grid(result_dir)

    snapshots_means = torch.load(pod_dir / "snapshots_means.pt", weights_only=True)


    
    dt = config._get_time_step(phy_dict['D'])
    result = romsolver._main_line(dt)

    result_dir.mkdir(exist_ok=True)
    torch.save(result, result_dir / f'DME_ROMresult.pt')

    Nh = Phi_r.shape[0]
    N3 = Nh // 8
    N = round(N3 ** (1/3))
    
    recon_rho = Phi_r @ result.cpu() + snapshots_means
    recon_rho = recon_rho.reshape(N, N, N, 8)

    torch.save(recon_rho, result_dir / 'DME_ROM_projection.pt')


    S_z = torch.tensor([0.5, 0.25, 0, -0.25, -0.5, 0.25, 0, -0.25])
    P = 2 * torch.einsum('i, klmi -> klm', S_z, recon_rho)
    P = P.cpu()
    torch.save(P, result_dir/'DME_ROM_P.pt')  




if __name__ == "__main__":
    main()

