from GraPL import GraPL_Segmentor, side_by_side, view_multichannel, PatchDL, GraPLNet, segment_voc
import glob
import tqdm
import matplotlib.pyplot as plt
from GraPL.evaluate import *
import numpy as np
import warnings
import torch
import pandas as pd
import os
warnings.filterwarnings("ignore")

with open("datasets/PascalVOC2012/VOC2012/ImageSets/Segmentation/val.txt", "r") as f:
    val_image_ids = f.read().split("\n")

params = {
    'iterations': 4,
    'k': 14,
    'd': 32,
    'lambda_': 64,
    'subset_size': 0.5,
    'max_epochs': 40,
    'min_epochs': 12,
    'n_filters': 32,
    'bottleneck_dim': 8,
    'compactness': 0.1,
    'sigma': 10,
    'seed': 0,
    'use_continuity_loss': True,
    'continuity_range': 1,
    'continuity_p': 1,
    'continuity_weight': 2,
    'use_min_loss': True,
    'use_coords': False,
    'use_embeddings': False,
    'use_color_distance_weights': True,
    'initialization_method': 'slic',
    'use_fully_connected': True,
    'use_collapse_penalty': False,
    'use_cold_start': False,
    'num_layers': 3,
    'use_graph_cut': True,
    'separate_noncontiguous_at_prediction': True,
    'separate_noncontiguous_at_training': False,
    'training_scale': 1.0
}

def run_experiment(base_params, num_trials=2, seeds=None):
    results_dir = f'experiment_results/voc_with_separation'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    aggregate_scores = {}
    trials = []
    mean_over_trials = {}
    if seeds is None:
        seeds = list(range(num_trials))
    print(f'Running experiment with seeds: {seeds}')
    with tqdm.tqdm(total=len(seeds) * len(val_image_ids)) as progress_bar:
        for trial_num in seeds:
            params = base_params.copy()
            params['seed'] = trial_num
            trial_results_dir = f'{results_dir}/{trial_num}'
            trial_scores = segment_voc(results_dir=trial_results_dir, debug_num=-1, resume=True, progress_bar=progress_bar, **params)
            image_ids = list(trial_scores.keys())
            metrics = trial_scores[image_ids[0]].keys()
            trial_scores = {metric: np.mean([trial_scores[id][metric] for id in image_ids]) for metric in metrics}
            trials.append(trial_scores)
    for metric in trials[0]:
        mean_over_trials[metric] = np.mean([trial[metric] for trial in trials])
    aggregate_scores = mean_over_trials
    return aggregate_scores


if __name__ == "__main__":
    # get arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="*")
    args = parser.parse_args()

    # run experiment
    run_experiment(params, num_trials=args.num_trials, seeds=args.seeds)