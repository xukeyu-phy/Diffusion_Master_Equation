import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path


def fit_curve(z, a, b):
    f =  - b * z + a 
    return f
def calculate_r_squared(fit_data, observed_data):
    residuals = observed_data - fit_data
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((observed_data - np.mean(observed_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

base_dir = Path(__file__).parent
folders = [f for f in base_dir.iterdir() if f.is_dir()]

plt.figure(figsize=(10, 6))

for folder in folders:
    xi = torch.load(folder/'xi.pt', weights_only=True)
    mesh = torch.load(folder/'Mesh.pt', weights_only=True)
    if xi.is_cuda: 
        xi = xi.cpu()
    if torch.is_tensor(mesh['grid_z']):
        mesh = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k,v in mesh.items()}
    
    with open(folder/'parameters.json','r') as f:
        phy_dict = json.load(f)
    R0 = phy_dict['R0']
    OD = phy_dict['OD']

    c = mesh['grid_x'].shape[2]//2
    z_axis = mesh['grid_z'][c, c, :].flatten()
    xi_z = xi[c, c, :].numpy().flatten()
    f_thero =  R0 - OD * z_axis
    # initial_guess = [R0, OD]
    initial_guess = [R0, OD]
    f = R0 * (xi_z - 0) + np.log(xi_z)

    params, _ = curve_fit(fit_curve, z_axis, f, p0=initial_guess)
    R0_fit, OD_fit = params
    f_fit = fit_curve(z_axis, R0_fit, OD_fit)

    params, _ = curve_fit(fit_curve, z_axis, f, p0=initial_guess)
    R0_fit, OD_fit = params
    f_fit = fit_curve(z_axis, R0_fit, OD_fit)

    # 计算 R² 值
    r_squared = calculate_r_squared(f_fit, f)

    # 输出拟合数据和 R²
    print(f'Fitting results for folder: {folder}')
    print(f'R0_fit: {R0_fit:.2f}, OD_fit: {OD_fit:.2f}')
    print(f'R²: {r_squared:.4f}')
    print('-' * 50)
    plt.plot(z_axis, f, '-', label=f'Simulated result : R0={R0:.2f}, OD={OD:.2f}, R²: {r_squared:.4f}')
    plt.plot(z_axis, f_thero, 'k--', linewidth=1)

plt.xlabel('z')
plt.ylabel(r'$B[\xi(z)]$')
plt.title('Curve Fitting of B[ξ(z)]')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=8)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(base_dir/'Bz_fit.png', dpi=300, bbox_inches='tight')
plt.show()

