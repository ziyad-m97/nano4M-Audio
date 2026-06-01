#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --time=04:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --output=notebook_job.out
#SBATCH --error=notebook_job.err
#SBATCH --partition=l40s

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm
cd /path/to/home/nano4M

python -m ipykernel install --user --name nanofm --display-name "nanofm"

python3 -c "
import json
with open('notebooks/COM304_FM_part4_nanoFlowMatching.ipynb', 'r') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
with open('notebooks/COM304_FM_part4_nanoFlowMatching.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print('Notebook outputs cleared')
"

export TMPDIR=/path/to/home/tmp
mkdir -p $TMPDIR

jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=nanofm notebooks/COM304_FM_part4_nanoFlowMatching.ipynb --output COM304_FM_part4_nanoFlowMatching.ipynb
