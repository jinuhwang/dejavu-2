from pathlib import Path
import argparse
import yaml
import pandas as pd

experiment_dir = Path('/workspace/configs/experiment')
data_path = Path('/workspace/configs/paths/data.yaml')
data_yaml = yaml.safe_load(data_path.read_text())
data_dir = Path(data_yaml['data_dir'])
log_dir = data_dir / 'logs' / 'train'

def populate_hard_configs(dataset: str, dry_run: bool):
    template_path = experiment_dir / f"{dataset}-hard-template.yaml"

    tmp = {} # We use dict so that more recent runs are prioritized
    for file in sorted(log_dir.glob('**/config_tree.log')):
        dataset_from_file = None
        dirpath_from_file = None
        target_reuse_rate_from_file = None
        for line in file.read_text().splitlines():
            if 'dataset:' in line:
                dataset_from_file = line.split('dataset:')[-1].strip()
            elif 'dirpath:' in line:
                dirpath_from_file = line.split('dirpath:')[-1].strip()
            elif 'target_reuse_rate:' in line:
                target_reuse_rate_from_file = line.split('target_reuse_rate:')[-1].strip()
            if all(v is not None for v in [dataset_from_file, dirpath_from_file, target_reuse_rate_from_file]):
                break
        if any(v is None for v in [dataset_from_file, dirpath_from_file, target_reuse_rate_from_file]):
            continue

        if dataset_from_file == dataset:
            tmp[(dataset, target_reuse_rate_from_file)] = dirpath_from_file

    template = template_path.read_text()

    for reuse_rate in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5]:
        config_path = experiment_dir / f"{dataset}-hard-{reuse_rate}.yaml"
        dirpath = tmp[(dataset, str(reuse_rate))]
        with open(config_path, "w") as f:
            if dry_run:
                print(f"Would save to {config_path} with target_reuse_rate={reuse_rate} and ckpt_path={dirpath}")
            else:
                f.write(template.format(target_reuse_rate=reuse_rate, ckpt_path=dirpath))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["msrvtt", "how2qa"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    populate_hard_configs(args.dataset, args.dry_run)
