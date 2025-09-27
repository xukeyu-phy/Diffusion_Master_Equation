
import json
import torch
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
from config import Config

current_dir = Path(__file__).parent
main_dir = Path(__file__).parent.parent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)


def rhs_GNL_deim(rho):
    config = Config(device, dtype)
    config._setup_matrices(1, 1)
    rho = rho.to(device)

    S_dot_rho = torch.einsum('i,klmi->klm', config.S_z, rho)
    S_dot_rho_expanded = S_dot_rho.unsqueeze(-1)
    A_SE_rho = torch.einsum('ij,klmj->klmi', config.A_SE, rho)
    rhs_GNL_deim = S_dot_rho_expanded * A_SE_rho
    return rhs_GNL_deim



def load_snapshots_and_parameters(dir, deim=False):

    folders = []
    for item in dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            folders.append(item)
    
    folders.sort(key=lambda x: int(x.name))    
    snapshots_list = []
    snapshots_deim_list = []
    parameters_list = []
    
    for i, folder in enumerate(folders):
        param_path = folder / "parameters.json"
        with open(param_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        parameters_list.append(params)
        
        data_path = folder / "data.pt"
        data = torch.load(data_path, map_location='cpu', weights_only=True)  
        if deim == True:
            Fh = rhs_GNL_deim(data)

        flatten_Fh_data = Fh.reshape(-1, 1)
        snapshots_deim_list.append(flatten_Fh_data)

        flattened_data = data.reshape(-1, 1)
        snapshots_list.append(flattened_data)


    valid_indices = [i for i, s in enumerate(snapshots_list) if s.numel() > 0]
    valid_snapshots = [snapshots_list[i] for i in valid_indices]
    snapshots_matrix = torch.cat(valid_snapshots, dim=1)

    valid_snapshots_deim = [snapshots_deim_list[i] for i in valid_indices]
    snapshots_deim_matrix = torch.cat(valid_snapshots_deim, dim=1)
    
    print(f"组装完成: snapshots矩阵形状 = {snapshots_matrix.shape}")
    print(f"矩阵大小: {snapshots_matrix.element_size() * snapshots_matrix.nelement() / 1024**2:.2f} MB")
    print(f"DEIM矩阵大小: {snapshots_deim_matrix.element_size() * snapshots_deim_matrix.nelement() / 1024**2:.2f} MB")
    return snapshots_matrix, snapshots_deim_matrix, parameters_list


def perform_svd(snapshots, k):
    print(f"输入矩阵形状: {snapshots.shape}")    
    start_time = time.time()
    snapshots = snapshots.cpu()
    snapshots_np = snapshots.numpy()    
    snapshots_means = np.mean(snapshots_np, axis=1)
    snapshots_means = snapshots_means.reshape(-1, 1)
    # snapshots_means[:, :] = 0
    snapshots_vibration = snapshots_np - snapshots_means
    U, S, Vt = np.linalg.svd(snapshots_vibration, full_matrices=False)

    U = torch.from_numpy(U)
    S = torch.from_numpy(S)
    Vt = torch.from_numpy(Vt)
    snapshots_means = torch.from_numpy(snapshots_means)
    
    if k is not None and k < len(S):
        U = U[:, :k]
        S = S[:k]
        Vt = Vt[:k, :]    
    end_time = time.time()
    print(f"SVD完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"奇异值数量: {len(S)}")
    print(f"前10个奇异值: {S[:10]}")    
    return U, S, Vt, snapshots_means

def perform_deim(snapshots, M, epsi):
    print(f"输入矩阵形状: {snapshots.shape}")    
    start_time = time.time()
    snapshots = snapshots.cpu()
    snapshots_np = snapshots.numpy() 
    U, S, Vt = np.linalg.svd(snapshots_np, full_matrices=False)  
 
    if epsi is True:
        M = np.where(np.cumsum(S ** 2) / np.sum(S ** 2) >= 1 - epsi)[0][0] + 1
    elif M is True:
        M = M
    elif M is None and epsi is None:
        M = 20

    Phi_F = U[:, :M]  # 取前 r 个 POD 模态
    # Phi_F = Phi_F / (np.linalg.norm(Phi_F, axis=0, keepdims=True) + 1e-12)
    P = []
    for k in range(M):
        if k == 0:
            p = np.argmax(np.abs(Phi_F[:, 0]))
        else:
            c = np.linalg.lstsq(Phi_F[P, :k], Phi_F[P, k], rcond=None)[0]
            r = Phi_F[:, k] - Phi_F[:, :k] @ c
            p = np.argmax(np.abs(r))
        P.append(p)
    P = np.array(P)
    end_time = time.time()
    print(f'DEIM Indices: {P}')
    print(f"SVD完成，耗时: {end_time - start_time:.2f} 秒")

    Phi_F = torch.from_numpy(Phi_F)
    P = torch.from_numpy(P)
    S = torch.from_numpy(S)
    Vt = torch.from_numpy(Vt)
    return Phi_F, P, S, Vt


def save_svd_results(U, S, Vt, dir, U_name, S_name, Vt_name, Mat=None, Mat_name=None):  
    torch.save(U, dir / U_name)
    torch.save(S, dir / S_name)
    torch.save(Vt, dir / Vt_name)

    if Mat is not None:
        torch.save(Mat, dir / Mat_name)
    print(f'SVD data has saved at {dir}')


def load_svd_results(dir, U_name, S_name, Vt_name):
    U = torch.load(dir / U_name, weights_only=True)
    S = torch.load(dir / S_name, weights_only=True)
    Vt = torch.load(dir / Vt_name, weights_only=True)    
    return U, S, Vt


def plot_sv(s, energy, vare, trun_modes, save_filename):
    plt_modes =20
    trun_modes = max( torch.argmax((energy >= 1-vare).int()).item() + 1, trun_modes)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Mode index i')
    ax1.set_ylabel('Singular value σᵢ', color=color)
    # ax1.bar(np.arange(1, num_modes + 1), s[:num_modes], color=color, alpha=0.6)
    ax1.semilogy(np.arange(1, plt_modes + 1), s[:plt_modes], 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(np.arange(1, plt_modes + 1))
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Create secondary axis for cumulative energy
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cumulative energy', color=color)
    ax2.plot(np.arange(1, plt_modes + 1), energy[:plt_modes], 's-', color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add truncation line if within plot range
    if trun_modes <= plt_modes:
        ax1.axvline(x=trun_modes + 0.5, color='k', linestyle='--', linewidth=1.5)
        ax1.text(trun_modes + 0.7, 0.9 * max(s[:plt_modes]),
                 f'Truncated at r={trun_modes}',
                 rotation=90, verticalalignment='top')
        trunc_sv = s[trun_modes - 1]
        ax1.text(trun_modes, trunc_sv * 0.3, f'$\\sigma_{trun_modes} = {trunc_sv:.2e}$', 
             fontsize=9, color='green', ha='center',
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
    ax1.text(1, s[0] * 0.3, f'$\\sigma_1 = {s[0]:.2e}$', 
         fontsize=9, color='blue', ha='center',
         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))


    # Add title and adjust layout
    plt.title(f'First {plt_modes} Singular Values and Cumulative Energy (Truncated at r={trun_modes})')
    fig.tight_layout()
    fig.savefig(save_filename, dpi=300)



def main(load=True):    

    hf_data_dir = main_dir / "data_temp_smallscale"  
    pod_dir = current_dir / "pod_result"
    pod_dir.mkdir(exist_ok=True)


    if load == True:
        Phi_h, S_h, Vt_h = load_svd_results(pod_dir, "Phi_h.pt", "S_h.pt", "Vt_h.pt")
        Phi_F, S_F, Vt_F = load_svd_results(pod_dir, "Phi_F.pt", "S_F.pt", "Vt_F.pt")
    else:
        snapshots, snapshots_deim, parameters = load_snapshots_and_parameters(hf_data_dir, deim=True)
        Phi_h, S_h, Vt_h, snapshots_means = perform_svd(snapshots, None)
        save_svd_results(Phi_h, S_h, Vt_h, pod_dir, "Phi_h.pt", "S_h.pt", "Vt_h.pt", snapshots_means, "snapshots_means.pt")

        Phi_F, P_F, S_F, Vt_F = perform_deim(snapshots_deim, None, None)      
        save_svd_results(Phi_F, S_F, Vt_F, pod_dir, "Phi_F.pt", "S_F.pt", "Vt_F.pt", P_F, "P_F.pt")
    
    total_energy = S_h.pow(2).sum()
    cumulative_energy = torch.cumsum(S_h.pow(2), dim=0) / total_energy    
    print("\n能量保留比例:")
    for i in [1, 2, 3, 4, 5, 10]:
        if i <= len(S_h):
            print(f"前{i}个模式保留能量: {cumulative_energy[i-1]:.6e}")    
    vare = 1e-6
    trun_modes = 5    
    plt_name = pod_dir / 'Phi_h.png'
    plot_sv(S_h, cumulative_energy, vare, trun_modes, plt_name)


    total_energy_F = S_F.pow(2).sum()
    cumulative_energy_F = torch.cumsum(S_F.pow(2), dim=0) / total_energy_F    
    print("\n能量保留比例:")
    for i in [1, 2, 3, 4, 5, 10]:
        if i <= len(S_F):
            print(f"前{i}个模式保留能量: {cumulative_energy_F[i-1]:.6e}")    
    vare_f = 1e-8
    trun_modes_f = 5
    plt_name_F = pod_dir / 'Phi_F.png'
    plot_sv(S_F, cumulative_energy_F, vare_f, trun_modes_f, plt_name_F)




if __name__ == "__main__":
    main(load=0)