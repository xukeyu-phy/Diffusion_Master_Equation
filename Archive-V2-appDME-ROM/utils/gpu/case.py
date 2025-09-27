import json
from pathlib import Path
import numpy as np
import shutil
from scipy.stats import qmc

class ParameterSampler:
    def __init__(self):
        # eps = 1e-15  # 极小的扰动，确保 l_bounds < u_bounds
        self.param_bounds = {
            'D': [1e-2, 0.3],     
            'eta': [1.0, 100],  
            'fD': [0.0, 0.6],  
            'R0': [1.0, 10.0],  
            'Qa': [0.2, 1.6], 
            'Qb': [0.2, 2.4],  
            'w': [0.17, 1.7],   
            'OD': [0.0, 10.0], 
        }
        self.param_names = list(self.param_bounds.keys())
        self.n_params = len(self.param_names)

    def _lhs_samples(self, n_samples=100, seed=1):
        sampler = qmc.LatinHypercube(d=self.n_params, seed=seed)
        sample = sampler.random(n=n_samples)
        
        low = np.array([bound[0] for bound in self.param_bounds.values()])
        high = np.array([bound[1] for bound in self.param_bounds.values()])        
        scaled_samples = qmc.scale(sample, low, high)
        
        param_samples = []
        for i in range(n_samples):
            param_dict = {self.param_names[j]: float(scaled_samples[i, j]) for j in range(self.n_params)}
            param_samples.append(param_dict)        
        return param_samples
    
    def _create_submit_script(self, folder_path, folder_index):
        script_content = f'''#!/bin/sh 
#SBATCH -p gpu1                # 使用gpu1队列
#SBATCH -N 1                   # 1个节点
#SBATCH --gpus-per-node=1      # 每个节点1个GPU
#SBATCH --cpus-per-gpu=1       # 每个GPU配1个CPU核
#SBATCH -o job.%j.out          # 输出文件
#SBATCH -e job.%j.err          # 错误文件
#SBATCH -J sub_{folder_index:04d}  # 作业名称

source /fs1/software/python/3.9_miniconda_4.10.3/etc/profile.d/conda.sh

# 激活 torch 环境
conda activate torch

# 可选：验证环境
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA availability: $(python -c 'import torch; print(torch.cuda.is_available())')"

# timeout 5m nvidia-smi dmon > nvi_1.log &
echo STARTED at `date`
yhrun python DME_main.py
echo FINISHED at `date`

echo "运行完成！"
'''
        
        script_file = folder_path / f"sub_{folder_index:04d}.sh"
        script_file.write_text(script_content)
        script_file.chmod(0o755)
        print(f"已创建提交脚本: {script_file}")
    
    def _copy_core_files(self, target_dir):
        current_dir = Path(__file__).parent
        core_dir = current_dir / "core"
        
        for item in core_dir.iterdir():
            if item.is_file():
                target_file = target_dir / item.name
                shutil.copy2(item, target_file)
                print(f"已复制: {item.name} -> {target_dir}")
            elif item.is_dir():
                target_subdir = target_dir / item.name
                shutil.copytree(item, target_subdir, dirs_exist_ok=True)
                print(f"已复制目录: {item.name} -> {target_dir}")        
        return True
    
    def _save_samples(self, samples, dir_path):        
        for i, sample in enumerate(samples):
            folder_name = f"{i:04d}"  
            folder_path = dir_path / folder_name
            folder_path.mkdir(exist_ok=True)
            
            param_file = folder_path / "parameters.json"
            param_file.write_text(json.dumps(sample, indent=2))            
            print(f"已保存参数到 {param_file}")
            
            self._copy_core_files(folder_path)
            self._create_submit_script(folder_path, i)
        
        return len(samples)
    
    def generate_parameter_space(self, dir_path, n_samples=500, seed=21):
        samples = self._lhs_samples(n_samples, seed)
        self._save_samples(samples, dir_path)        
        return samples




if __name__ == "__main__":
    sampler = ParameterSampler()
    current_dir = Path(__file__).parent
    samples = sampler._lhs_samples(n_samples=1000, seed=21)
    num = sampler._save_samples(samples, current_dir)

    print(f"总共保存了 {num} 个样本到当前目录")


    max_jobs_per_script = 100
    num_scripts = (num + max_jobs_per_script - 1) // max_jobs_per_script
    
    for script_index in range(num_scripts):
        file_name = current_dir / f'submit_jobs_{script_index+1:02d}.sh'
        with open(file_name, "w") as file:
            file.write("#!/bin/bash\n")
            file.write("# 自动生成的批量提交脚本\n")
            file.write(f"# 批次 {script_index+1}/{num_scripts}\n")
            file.write("# 使用方法: bash submit_jobs_XX.sh\n")
            file.write("# 注意: 请确保在当前目录运行\n\n")
            
            start_index = script_index * max_jobs_per_script
            end_index = min((script_index + 1) * max_jobs_per_script, num)
            
            for ii in range(start_index, end_index):
                folder_name = f"{ii:04d}"
                file.write(f'echo "提交作业 {folder_name}"\n')
                file.write(f'cd {folder_name}\n')
                file.write(f'yhbatch sub_{folder_name}.sh\n')
                file.write(f'cd ..\n')
                file.write(f'sleep 1  # 稍微延迟一下，避免提交过于频繁\n')
                file.write(f'echo ""\n')
        
        Path(file_name).chmod(0o755)
        print(f"已创建提交脚本: {file_name}")