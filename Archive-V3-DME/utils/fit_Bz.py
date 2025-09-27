__all__ = ['device', 'dtype']

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# 定义拟合函数
def fit_curve(z, a, b):
    f =   - b * z
    return f



def main():
    current_dir = Path(__file__).parent

    xi = torch.load(current_dir/'xi.pt', weights_only=True)
    mesh = torch.load(current_dir/'Mesh.pt', weights_only=True)
    rho = torch.load(current_dir/'DME_HFresult.pt', weights_only=True)
    Srho = torch.load(current_dir/'P_Szrho.pt', weights_only=True)

    with open(current_dir / 'parameters.json', 'r') as f:
        phy_dict = json.load(f)
    
    R0 = phy_dict['R0']
    OD = phy_dict['OD']
    
    c = mesh['grid_x'].shape[2]//2
    z_axis = mesh['grid_z'][c, c, :].cpu().numpy().flatten()
    xi_z = xi[c, c, :].cpu().numpy().flatten()
    rho_z = rho[c, c, :].cpu().numpy().flatten()
    Srho = Srho[c, c, :].cpu().numpy().flatten()
    # z_axis = z_axis[ :90]
    # xi_z = xi_z[: 90]
    f_thero = 0 - OD * z_axis

    f = R0 * (xi_z - 1) + np.log(xi_z)
    initial_guess = [R0, OD]

    params, covariance = curve_fit(fit_curve, z_axis, f, p0=initial_guess)
    a_fit, b_fit = params
    
    f_fit = fit_curve(z_axis, a_fit, b_fit)
    

    residuals = f - f_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((f - np.mean(f))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    param_text = f'Fitted value:\n OD = {b_fit:.4f} \n R² = {r_squared:.4f}'
    theo_text = f'Theoretical value:\n OD = {OD:.4f}'
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(z_axis, f, 'b-', label='Original')
    plt.plot(z_axis, f_fit, 'r--', label='Fitted')
    plt.plot(z_axis, f_thero, 'k-', label='therotical')
    plt.xlabel('z')
    plt.ylabel('B[ξ(z)]')
    plt.title('Curve Fitting of B[ξ(z)]')
    plt.legend()
    plt.grid(True)
    
    plt.text(0.02, 0.50, param_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.text(0.98, 0.50, theo_text, transform=plt.gca().transAxes,
            verticalalignment='center', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    
    plt.savefig(current_dir / 'fit_Bz.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f'Fitted parameters: OD = {b_fit:.4f}, R0 = {a_fit:.4f}')
    print(f'Theoretical values: OD = {OD:.4f}, R0 = {R0:.4f}')
    print(f'R² = {r_squared:.4f}')
        


if __name__ == "__main__":
    main()