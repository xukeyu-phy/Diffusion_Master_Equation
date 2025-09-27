
__all__ = ['device', 'dtype']


import json
import torch
from typing import Dict, Any, Optional, Tuple, List, Union
from scipy.stats import qmc  
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# from config import Config



main_dir = Path(__file__).parent.parent
current_dir = Path(__file__).parent
current_dir = current_dir/ 'result'
xi = torch.load(current_dir/'xi.pt', weights_only=True)
mesh = torch.load(current_dir/'Mesh.pt', weights_only=True)
P_Szrho = torch.load(current_dir/'P_Szrho.pt', weights_only=True)
rho = torch.load(current_dir / f'DME_HFresult.pt', weights_only=True)
x = mesh['grid_x']
y = mesh['grid_y']
z = mesh['grid_z']
coordi = mesh['grid']

param_filename = current_dir.parent / 'parameters.json'
with open(param_filename, 'r') as f:
    phy_dict = json.load(f)
# config = Config(device, dtype)
R0 = phy_dict['R0']
OD = phy_dict['OD']

c = 45
z_axis = z[c, c, :].cpu().numpy()
xi_z = xi[c, c, :].cpu().numpy()
# fig = plt.plot(figsize = (12,8))
fig = plt.plot(z_axis, xi_z)
plt.xlabel('z')
plt.ylabel(r'$\xi(r)$')
plt.grid(True)
plt.savefig(current_dir/'xi_z.png')


