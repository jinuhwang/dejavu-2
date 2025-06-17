import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml

import matplotlib
import numpy as np
import yaml
from matplotlib.patches import Patch

import sys
if '/workspace' not in sys.path:
    sys.path.append('/workspace')

from src.utils.flops import get_cmc_flops, get_eventful_flops, get_orginal_vit_flops, get_reusevit_flops, get_diffrate_flops

font_size = 14
plt.rcParams["font.family"] = 'HelveticaNeue'
matplotlib.rcParams['font.size'] = font_size
colors = yaml.safe_load(open("/workspace/scripts/colors.yaml"))

# First, gather the data
tempclip_path = Path('/workspace/third_parties/NExT-GQA/code/TempGQA')
with open('/workspace/configs/paths/data.yaml') as f:
    data_dir = Path(yaml.safe_load(f)['data_dir'])
run_dir = data_dir / 'logs/eval/runs'

tmp = {} # So that more recent runs are prioritized
globs = []
for file in sorted(run_dir.glob('**/hparams.yaml')):
    globs.append(file)


for file in tqdm(globs, desc='Parsing results'):
    d = {}
    
    metrics_path = file.parent / 'metrics.csv'
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        if 'test/reuse_rate_mean' in df.columns:
            d['reuse_rate_mean'] = df.iloc[-1]['test/reuse_rate_mean']
        else:
            continue
    
    with file.open() as f:
        y = yaml.safe_load(f)
        try:
            dataset_str = y['data']['dataset_str']
            test_split = y['data']['test_split']
            run_name = y['model']['reuse_model_name']
        except:
            continue

    tmp[(dataset_str, run_name, test_split)] = d
    
# Make run_name and test_split
for k, v in tmp.items():
    dataset_str, run_name, test_split = k
    v['dataset_str'] = dataset_str
    v['run_name'] = run_name
    v['test_split'] = test_split
    
reuse_rate_df = pd.DataFrame(list(tmp.values()))
reuse_rate_df

def parse_run_name(run_name):
    d = {}
    run_name = run_name[run_name.find('log_')+4:]
    if 'reuse' in run_name:
        run_name = run_name.lstrip('reuse-')
        type_name = 'ReuseViT'
    elif 'diffrate' in run_name:
        type_name = 'DiffRate'
        run_name = run_name.split('_')[1]
    elif 'cmc' in run_name: 
        type_name = 'CMC'
        run_name = run_name.split('_')[1]
    elif 'eventful' in run_name:
        type_name = 'Eventful'
        run_name = run_name.split('_')[1]
    elif 'original' in run_name:
        type_name = 'Original'
        run_name = 'Original'
    else:
        raise ValueError(f'Unknown run name: {run_name}')

    d['name'] = run_name
    d['type'] = type_name

    return d

# Parse result based on stdout of TempClip
def get_tempclip_results(lines):
    d = {}

    idx = 0
    while idx < len(lines):
        if lines[idx].strip() == '=======merge post-hoc and gauss mask======':
            line = lines[idx + 2]
            splitted = line.split()
            d['Acc&GQA'] = splitted[0]
            d['mIoP'] = splitted[1]
            d['TIoP@0.3'] = splitted[2]
            d['TIoP@0.5'] = splitted[3]
            d['mIoU'] = splitted[4]
            d['TIoU@0.3'] = splitted[5]
            d['TIoU@0.5'] = splitted[6]
            break
        idx += 1

    return d

for result_dir, get_result_fn, test_split, csv_path in (
    (tempclip_path, get_tempclip_results, 'test', 'tempclip.csv'),
):
    ret = []
    target_reuse_rate_df = reuse_rate_df[reuse_rate_df['test_split'] == test_split]
    for file in result_dir.glob('*.txt'):
        run_name = file.stem
        d = parse_run_name(run_name)
        with file.open() as f:
            lines = f.readlines()
            d.update(get_result_fn(lines))
        if d['type'] == 'reuse':
            try:
                reuse_rate = target_reuse_rate_df[target_reuse_rate_df['run_name'] == d['name']]['reuse_rate_mean'].item()
                d['reuse_rate_mean'] = reuse_rate
            except KeyError:
                pass
            except ValueError:
                print(f'{d["name"]} not found in reuse_rate_df')
        ret.append(d)

    df = pd.DataFrame(ret)
    df.to_csv(csv_path, index=False)

# 2. Calculate FLOPs based on Reuse Rate
# Calculate FLOPs from reuse rate
model_size = 'large'
for _, row in df.iterrows():

    try:    
        if row['type'] == 'CMC':
            avg_reuse_ratio = float(row['reuse rate'])
            flops = get_cmc_flops(
                model_size,
                avg_reuse_ratio,
                reuse_start_before_mlp=True
            )
            df.at[row.name, 'flops'] = flops
        elif row['type'] == 'Eventful':
            r = int(row['target'])
            flops = get_eventful_flops(model_size, r)
            df.at[row.name, 'flops'] = flops
        elif 'ReuseViT' in row['type']:
            # will be fixed later
            print(row['dataset'], row['flops'])
            reuse_rates = [float(row['reuse rate'])] * 4
            flops = get_reusevit_flops(model_size, reuse_rates)
            print(row['dataset'], flops)
            df.at[row.name, 'flops'] = flops
        elif row['type'] == 'Original':
            original_flops = get_orginal_vit_flops(model_size)
            df.at[row.name, 'flops'] = original_flops
        elif row['type'] == 'DiffRate':
            flops = float(row['target'])
            flops = get_diffrate_flops(model_size, flops)
            df.at[row.name, 'flops'] = flops
    except Exception as e:
        print(f"Error at {row.name}({row['type']}): {e}")

df['reduction'] = df['flops'] / original_flops

# 3. Plot FLOPs reduction
xlabel_name = 'Grounding Accuracy (%)'
ylabel_names = 'Reduction'

dotsize = 60
linewidth = 3
markersize = 15

fig, ax = plt.subplots(figsize=(4.5, 2.25))

# -- Tick parameters --
ax.tick_params(axis='x', left=False, bottom=True, pad=-1)
ax.tick_params(axis='y', left=True, bottom=False, pad=-2)

# -- Add horizontal grid lines at each y-tick --
for ytick in ax.get_yticks():
    ax.axhline(y=ytick, color='grey', linewidth=0.5, alpha=0.5)

# -- Plot the "No Reuse" star --
original_accs = df[(df['type'] == 'Original')]['Acc&GQA'].iloc[0]
ax.scatter([original_accs], [1.0],
           color='black',
           s=dotsize + 10,
           zorder=10,
           marker="*")

# -- Plot each data series from datas[row_idx][col_idx] --
labels = ['ReuseViT', 'CMC', 'Eventful', 'DiffRate']

for label in labels:
    x_arr = df[df['type'] == label]['Acc&GQA'].tolist()
    y_arr = df[df['type'] == label]['reduction'].tolist()
    ax.plot(x_arr, y_arr,
            linestyle='-',
            color=colors[label],
            linewidth=linewidth,
            zorder=3,
            marker='.',
            markersize=markersize)

# Save this subplot to a file.
outfile = f"nextqa.png"
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close(fig)