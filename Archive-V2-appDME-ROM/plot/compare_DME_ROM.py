import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple, Optional
# from config import device, dtype 

current_dir = Path(__file__).parent
pod_dir = current_dir / 'pod_results'
result_dir = current_dir / "result"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)


recon_rho = torch.load(result_dir/'DME_ROM_projection.pt', weights_only=True, map_location=device)
hf_rho = torch.load(result_dir/'DME_HFresult.pt', weights_only=True, map_location=device)


recon_rho = recon_rho.reshape(-1,1)
hf_rho = hf_rho.reshape(-1,1)

# abs_tol = 1e-5
# for i in torch.abs(recon_rho - hf_rho):
#     if i < abs_tol:
#         i = 0.0
relative_error = torch.abs(recon_rho - hf_rho) / torch.abs(hf_rho + 1e-8)
abs_error = torch.abs(recon_rho - hf_rho)
fig, ax1 = plt.subplots(figsize=(12, 6))


line1 = ax1.plot(hf_rho, 'b-', label='HF_result', linewidth=2, alpha=0.8)

line2 = ax1.plot(recon_rho, 'r--', label='ROM_result_compute', linewidth=2, alpha=0.8)


ax1.set_xlabel('Index')
ax1.set_ylabel('Value', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)


ax2 = ax1.twinx()
line3 = ax2.plot(abs_error, 'g-', label='Absolute Error', linewidth=1, alpha=0.6)
ax2.set_ylabel('Absolute Error', color='green')
ax2.tick_params(axis='y', labelcolor='green')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title('Comparison: HF_result and ROM_projection with Absolute Error')
plt.tight_layout()
fig.savefig(result_dir/'comparison_plot.png', dpi=300)

