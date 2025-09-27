
__all__ = ['device', 'dtype']


import json
import torch
from pathlib import Path
import numpy as np


from config import Config
from DME_solver import DMESolver

main_dir = Path(__file__).parent.parent
current_dir = Path(__file__).parent      
result_dir = current_dir / "result"
result_dir.mkdir(exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)


def main():

    param_filename = current_dir / 'parameters.json'
    with open(param_filename, 'r') as f:
        phy_dict = json.load(f)
    dmesolver = DMESolver(device, dtype, phy_dict, result_dir)
    config = Config(device, dtype)
    config._create_non_uniform_grid(result_dir)

    dmesolver._init_phy_ps(phy_dict)
    dt = config._get_time_step(phy_dict['D']) 
    result, xi = dmesolver._main_line(dt)
    torch.save(result, result_dir / f'DME_HFresult.pt')
    config._setup_matrices(phy_dict['Qa'], phy_dict['Qb'])
    S_z = config.S_z
    P_Szrho = 2 * torch.einsum('i, klmi -> klm', S_z, result)  # Calculate the electric polarization rate

    torch.save(P_Szrho, result_dir/'P_Szrho.pt')    
    torch.save(xi, result_dir/'xi.pt')




if __name__ == "__main__":
    main()
