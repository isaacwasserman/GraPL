import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import sklearn.metrics as metrics
import skimage.metrics
from torchmetrics import JaccardIndex
import torch
import glob
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
from PIL.PngImagePlugin import PngInfo

def hungarian_match(preds, targets, ignore_indices=[]):
    flat_preds = preds.flatten()
    flat_targets = targets.flatten()
    num_samples = flat_targets.shape[0]

    pred_k = flat_preds.max() + 1
    target_k = flat_targets.max() + 1

    max_k = max(pred_k, target_k)
    num_correct = np.zeros((max_k, max_k))

    for pred_segment_index in range(pred_k):
        for target_segment_index in range(target_k):
            # if target_segment_index in ignore_indices:
            #     continue
            votes = int(((flat_preds == pred_segment_index) * (flat_targets == target_segment_index)).sum())
            num_correct[pred_segment_index, target_segment_index] = votes

    match = linear_assignment(num_samples - num_correct)
    match = np.stack(match, axis=1)
    return match

def match_segmentations(seg, gt, ignore_indices=[]):
    seg_values = np.unique(seg)
    for i,v in enumerate(seg_values):
        seg[seg == v] = i
    match = hungarian_match(seg, gt, ignore_indices=ignore_indices)
    matched_seg = np.full_like(seg, 0)
    already_matched = []
    for i in range(max(seg.max(), gt.max()) + 1):
        m = match[i, 1]
        if m not in already_matched:
            matched_seg[seg == i] = m
            already_matched.append(m)
        else:
            print("Duplicate match found")
    return matched_seg

def match_size(preds, targets):
    preds_tensor = torch.tensor(preds).float().unsqueeze(0).unsqueeze(0)
    matched = torch.nn.functional.interpolate(preds_tensor, size=targets.shape, mode='nearest').squeeze(0).squeeze(0).long().numpy()
    return matched

def match(seg, targets, size=True, indices=True):
    if size:
        seg = match_size(seg, targets)
    if indices:
        seg = match_segmentations(seg, targets.astype(int))
    return seg

def f1_score(seg, targets, resize=True, match_segments=True):
    seg = match(seg, targets, size=resize, indices=match_segments)
    return metrics.f1_score(seg.flatten(), targets.flatten(), average='weighted')

def accuracy(seg, targets, resize=True, match_segments=True):
    seg = match(seg, targets, size=resize, indices=match_segments)
    return metrics.accuracy_score(seg.flatten(), targets.flatten())

def jaccard(seg, targets, resize=True, match_segments=True, ignore_indices=[]):
    seg = match(seg, targets, size=resize, indices=match_segments)
    jaccard = JaccardIndex(num_classes=max(len(np.unique(seg)),len(np.unique(targets))), task="multiclass", ignore_index=0, average='weighted')
    seg = torch.tensor(seg).unsqueeze(0).unsqueeze(0)
    targets = torch.tensor(targets).unsqueeze(0).unsqueeze(0)
    score = jaccard(seg, targets).item()
    return score
    # return metrics.jaccard_score(seg.flatten(), targets.flatten(), average='weighted')

def v_measure(seg, targets, resize=True, match_segments=True):
    seg = match(seg, targets, size=resize, indices=match_segments)
    return metrics.v_measure_score(seg.flatten(), targets.flatten())

def calculate_overlap(r1, r2):
    # intersection
    a = np.count_nonzero(r1 * r2)
    # union
    b = np.count_nonzero(r1 + r2)
    
    return a/b

def segmentation_covering(seg, targets, resize=True, match_segments=True):
    seg = match(seg, targets, size=resize, indices=match_segments)
    segmentation1 = seg
    segmentation2 = targets
    N = segmentation1.shape[0] * segmentation1.shape[1]
    
    maxcoverings_sum = 0
    
    # Sum over regions
    for label1 in np.unique(segmentation1):
        # where the segmentation has a specific label
        region1 = (segmentation1 == label1).astype(int) 
        # |R| is the size of nonzero elements as this the region size
        len_r = np.count_nonzero(region1) 
        max_overlap = 0
        # Calculate max overlap 
        for label2 in np.unique(segmentation2):
            # where the segmentation has a specific label
            region2 = (segmentation2 == label2).astype(int)
            # Calculate overlap
            overlap = calculate_overlap(region1, region2)
            max_overlap = max(max_overlap, overlap)
        
        maxcoverings_sum += (len_r * max_overlap)
    sc = (1 / N) * maxcoverings_sum
    return sc

def variation_of_information(seg, targets, resize=True, match_segments=True):
    seg = match(seg, targets, size=resize, indices=match_segments)
    vi = skimage.metrics.variation_of_information(seg, targets)
    return vi

def probabalistic_rand_index(seg, targets, resize=True, match_segments=True):
    seg = match(seg, targets, size=resize, indices=match_segments)
    seg = seg.flatten()
    targets = targets.flatten()
    pri = metrics.rand_score(targets, seg)
    return pri


def make_valid_segmentation(a):
    if len(a.shape) == 3:
        shape = a.shape[:2]
        a = a.reshape(-1, a.shape[2])
        colors = np.unique(a, axis=0)
        a = a.reshape(shape[0], shape[1], a.shape[1])
        new_a = np.zeros(shape)
        for i,color in enumerate(colors):
            new_a[(a == color).all(axis=2)] = i
        a = new_a
    elif a.dtype != int:
        a /= a.max()
        a *= 255
        a = a.astype(int)
    return a

def proportional_potts_energy(seg):
    padded_seg = np.pad(seg, 1, mode='constant', constant_values=-1)
    left = padded_seg[:, :-1]
    right = padded_seg[:, 1:]
    top = padded_seg[:-1, :]
    bottom = padded_seg[1:, :]
    horizontal_valid = (left != -1) & (right != -1)
    vertical_valid = (top != -1) & (bottom != -1)
    horizontal = (left != right)[horizontal_valid].sum()
    vertical = (top != bottom)[vertical_valid].sum()
    w = seg.shape[-1]
    n_edges = 2*(w**2) - 2*w
    return (horizontal + vertical) / n_edges

def calc_k(seg):
    k = len(np.unique(seg))
    return k

def calc_k_error(seg, gt):
    # calculates percent error in k. Can be negative
    k_hat = calc_k(seg)
    k = calc_k(gt)
    return (k - k_hat) / k

def calc_delta_k(seg, k_initial):
    k = calc_k(seg)
    return k - k_initial

def bsds_score(id, segmentation_, return_mean=True):
    save_scores = False
    if isinstance(segmentation_, str):
        segmentation = Image.open(segmentation_)
        metadata = segmentation.text
        segmentation = np.array(segmentation)
        save_scores = True
    else:
        segmentation = segmentation_
        metadata = None
    
    segmentation = make_valid_segmentation(segmentation)
    gt_paths = glob.glob(f"datasets/BSDS500/gt/{id}-*.csv")
    gts = [np.genfromtxt(path, delimiter=',').astype(int) for path in gt_paths]
    segmentation = match(segmentation, gts[0], size=True, indices=False)
    scores = {"f1_score": [], "accuracy": [], "jaccard": [], "v_measure": [], "segmentation_covering": [], "variation_of_information": [], "probabalistic_rand_index": [], "proportional_potts_energy": [], "delta_k": [], "k_error": [], "training_time": [], "prediction_time": [], "total_time": []}
    
    already_scored = False
    if metadata is not None:
        already_scored = True
        for key in scores.keys():
            if key not in metadata:
                already_scored = False
                break
            else:
                scores[key].append(float(metadata[key]))
    
    if not already_scored:
        for gt in gts:
            segmentation_matched = match(segmentation, gt, size=False, indices=True)
            scores["f1_score"].append(f1_score(segmentation_matched, gt, resize=False, match_segments=False))
            scores["accuracy"].append(accuracy(segmentation_matched, gt, resize=False, match_segments=False))
            scores["jaccard"].append(jaccard(segmentation_matched, gt, resize=False, match_segments=False))
            scores["v_measure"].append(v_measure(segmentation_matched, gt, resize=False, match_segments=False))
            scores["segmentation_covering"].append(segmentation_covering(segmentation_matched, gt, resize=False, match_segments=False))
            scores["variation_of_information"].append(variation_of_information(segmentation_matched, gt, resize=False, match_segments=False))
            scores["probabalistic_rand_index"].append(probabalistic_rand_index(segmentation_matched, gt, resize=False, match_segments=False))
            scores["k_error"].append(calc_k_error(segmentation_matched, gt))
        scores["proportional_potts_energy"].append(proportional_potts_energy(segmentation_matched))
        if metadata is not None and "k" in metadata:
            scores["delta_k"].append(calc_delta_k(segmentation_matched, int(metadata["k"])))
            scores["training_time"].append(float(metadata["training_time"]))
            scores["prediction_time"].append(float(metadata["prediction_time"]))
            scores["total_time"].append(float(metadata["total_time"]))

    if return_mean:
        for key in scores.keys():
            scores[key] = np.mean(scores[key])

    if not already_scored:
        if save_scores:
            new_metadata = PngInfo()
            for key in metadata:
                new_metadata.add_text(key, metadata[key])
            for key in scores:
                new_metadata.add_text(key, str(scores[key]))
            segmentation = Image.open(segmentation_)
            segmentation.save(segmentation_, pnginfo=new_metadata)
    return scores

def voc_score_both_tasks(id, seg_path, return_mean=True):
    object_scores = voc_score(id, seg_path, segmentation_mode="SegmentationObject", return_mean=return_mean)
    class_scores = voc_score(id, seg_path, segmentation_mode="SegmentationClass", return_mean=return_mean)
    combined_scores = {}
    accuracy_metrics = ["f1_score", "accuracy", "jaccard", "v_measure", "segmentation_covering", "variation_of_information", "probabalistic_rand_index", "proportional_potts_energy"]
    for key in object_scores:
        if key in accuracy_metrics:
            combined_scores[key + "_object"] = object_scores[key]
        else:
            combined_scores[key] = object_scores[key]
    for key in class_scores:
        if key in accuracy_metrics:
            combined_scores[key + "_class"] = class_scores[key]
    return combined_scores



def voc_score(id, seg_path, segmentation_mode="SegmentationObject", return_mean=True):
    # Get segmentation and metadata from file
    segmentation = Image.open(seg_path)
    metadata = segmentation.text
    segmentation = np.array(segmentation)
    segmentation = make_valid_segmentation(segmentation)
    
    # Reformat ground truth
    gt = plt.imread(f"datasets/PascalVOC2012/VOC2012/{segmentation_mode}/{id}.png")
    colors = np.unique(gt.reshape(-1, gt.shape[2]), axis=0)
    ignore_mask = np.isclose(gt, [0.8784314, 0.8784314, 0.7529412, 1.       ], atol=0.01).all(axis=2)
    background_mask = np.isclose(gt, [0, 0, 0, 1], atol=0.01).all(axis=2)
    object_colors = colors[~np.isclose(colors, [0, 0, 0, 1], atol=0.01).all(axis=1)]
    object_colors = object_colors[~np.isclose(object_colors, [0.8784314, 0.8784314, 0.7529412, 1], atol=0.01).all(axis=1)]
    integer_gt = np.zeros_like(gt[:, :, 0])
    integer_gt[ignore_mask] = 0
    integer_gt[background_mask] = 1
    for i in range(len(object_colors)):
        mask_index = i + 2
        integer_gt[np.isclose(gt, object_colors[i], atol=0.01).all(axis=2)] = mask_index
    gt = integer_gt
    
    # Match and score
    segmentation = match(segmentation, gt, size=True, indices=True)
    scores = {"jaccard": [], "training_time": [], "prediction_time": [], "total_time": []}
    scores["jaccard"].append(jaccard(segmentation, gt, resize=False, match_segments=False, ignore_indices=[0]))
    if metadata is not None and "k" in metadata:
        scores["training_time"].append(float(metadata["training_time"]))
        scores["prediction_time"].append(float(metadata["prediction_time"]))
        scores["total_time"].append(float(metadata["total_time"]))
    if return_mean:
        for key in scores.keys():
            scores[key] = np.mean(scores[key])
    new_metadata = PngInfo()
    for key in metadata:
        new_metadata.add_text(key, metadata[key])
    for key in scores:
        new_metadata.add_text(key, str(scores[key]))
    segmentation = Image.open(seg_path)
    segmentation.save(seg_path, pnginfo=new_metadata)
    return scores

def get_score_means(scores):
    mean_scores = {}
    for id in scores:
        for metric in scores[id]:
            if metric not in mean_scores:
                mean_scores[metric] = []
            mean_scores[metric].append(scores[id][metric])
    for metric in mean_scores:
        mean_scores[metric] = np.mean(mean_scores[metric])
    return mean_scores

def bsds_score_directory(directory):
    segmentation_paths = glob.glob(directory)
    scores = {}
    for path in tqdm.tqdm(segmentation_paths):
        seg = plt.imread(path)
        id = path.split("/")[-1].split(".")[0]
        scores[id] = bsds_score(id, seg)
        # print(scores[id]["v_measure"])
    mean_scores = get_score_means(scores)
    return mean_scores
