import torch
torch.use_deterministic_algorithms(True)
import numpy as np
import time
from PIL import Image
import skimage.segmentation
import skimage.color
import matplotlib.pyplot as plt
import pandas as pd
import gco
from GraPL.evaluate import bsds_score, voc_score
import torchvision
from sklearn.decomposition import KernelPCA, PCA
from sklearn.cluster import KMeans
import glob
import copy
from PIL.PngImagePlugin import PngInfo
import matplotlib.cm as colormap
import matplotlib.colors
import json
import os
import tqdm

# Find fastest device available
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

if device == "mps":
    from torch import mps
    backend = mps
elif device == "cuda":
    from torch import cuda
    backend = cuda

dinov2 = None

def initialize_dino():
    global dinov2
    if dinov2 is None:
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        dinov2.to(device)
        dinov2.eval()

def epoch_schedule(x_, max_epochs=40, min_epochs=10, n_iter=4):
    """Calculates the number of epochs to train for based on the current iteration

    Args:
        x_ (int): current iteration
        max_epochs (int, optional): maximum number of epochs to train for. Defaults to 40.
        min_epochs (int, optional): minimum number of epochs to train for. Defaults to 10.
        n_iter (int, optional): total number of iterations to train for. Defaults to 4.

    Returns:
        int: number of epochs to train for
    """
    x = x_
    if x == 0:
        return max_epochs
    v = ((5*n_iter)/x) + min_epochs
    v = min(round(v), max_epochs)
    return v

def initial_labels(image, d, n_segments, compactness=10, sigma=1, method='slic'):
    """Generates patch-level labels for an image using SLIC

    Args:
        image (ndarray): image represented by (H,W,C) array
        d (int): number of patches per dimension to assign labels to; function will return d^2 labels
        n_segments (int): number of segments to generate
        compactness (int, optional): compactness parameter passed to SLIC. Defaults to 10.
        sigma (int, optional): sigma parameter passed to SLIC. Defaults to 1.

    Returns:
        tensor: softmaxed labels for each patch in the image (d^2, n_segments)
        ndarray: the full resolution segmentation
    """
    if method == 'random':
        labels = torch.randint(0, n_segments, (d*d,)).unsqueeze(0)
        seg = torch.nn.functional.interpolate(labels.float().reshape(d,d).unsqueeze(0).unsqueeze(0), size=(image.shape[0], image.shape[1]), mode='nearest').squeeze()
        labels = torch.nn.functional.one_hot(labels, n_segments).squeeze()
    elif method == 'knn':
        # pick n_segments random points in dxd grid
        x = torch.randint(0, d, (n_segments, 2))
        labels = torch.zeros((d,d)).long()
        for i in range(d):
            for j in range(d):
                distances = torch.sqrt((x[:,0] - i)**2 + (x[:,1] - j)**2)
                labels[i,j] = torch.argmin(distances)
        labels = labels.reshape(d*d)
        seg = torch.nn.functional.interpolate(labels.float().reshape(d,d).unsqueeze(0).unsqueeze(0), size=(image.shape[0], image.shape[1]), mode='nearest').squeeze()
        labels = torch.nn.functional.one_hot(labels, n_segments).squeeze()
    elif method == 'kmeans':
        # create array of all locations in dxd grid using meshgrid
        x = torch.arange(0, d)
        x = torch.meshgrid(x, x, indexing='ij')
        x = torch.stack(x, dim=-1).reshape(-1, 2)
        # run kmeans on the grid
        kmeans = KMeans(n_clusters=n_segments, random_state=0, n_init='auto').fit(x)
        labels = torch.tensor(kmeans.labels_).type(torch.int64)
        seg = torch.nn.functional.interpolate(labels.float().reshape(d,d).unsqueeze(0).unsqueeze(0), size=(image.shape[0], image.shape[1]), mode='nearest').squeeze()
        labels = torch.nn.functional.one_hot(labels, n_segments).squeeze()

    else:
        seg = skimage.segmentation.slic(image,
                                        n_segments=n_segments, compactness=compactness, sigma=sigma,
                                        enforce_connectivity=False, convert2lab=True)

        t = torch.tensor(seg).unsqueeze(0).unsqueeze(0).float()
        # bin the image
        kernel_width = image.shape[0] // d
        kernel_height = image.shape[1] // d
        regions = torch.nn.functional.unfold(t, (kernel_width, kernel_height), stride=(kernel_width, kernel_height), padding=0)
        regions = regions.permute(0,2,1).squeeze().to(torch.int64).squeeze(0)
        # count occurences of each segment in each bin
        labels = torch.nn.functional.one_hot(regions, int(regions.max()) + 1).float()
        labels = torch.sum(labels, dim=1)
        labels = torch.nn.functional.softmax(labels, dim=1)
    return labels, seg

def get_uniform_smoothness_pw_single_image(img_shape):
    """Returns the edges and weights for a grid graph with uniform smoothness

    Args:
        img_shape (tuple): shape of the image represented by a (H,W) tuple
        
    Returns:
        list: [edges, edge_weights] where edges is a (E,2) array of edges and edge_weights is a (E,) array of edge weights equal to 1
    """
    H, W = img_shape
    E = (H - 1) * W + H * (W - 1)

    edges = np.empty((E, 2), dtype=int)
    edge_weights = np.ones(E, dtype=np.single)
    idx = 0

    # horizontal edges
    for row in range(H):
        edges[idx:idx+W-1,0] = np.arange(W-1) + row * W
        edges[idx:idx+W-1,1] = np.arange(W-1) + row * W + 1
        idx += W-1

    # vertical edges
    for col in range(W):
        edges[idx:idx+H-1,0] = np.arange(0, (H-1)*W, W) + col
        edges[idx:idx+H-1,1] = np.arange(W, H*W, W) + col
        idx += H-1

    return [edges, edge_weights]

def graph_cut(probabilities, d, k, lambda_):
    """Performs graph cut on a single image based only on provided unary potentials in the form of probabilities

    Args:
        probabilities (ndarray): unary potentials for each pixel in the image represented by a (d*d, n_segments) array
        d (int): number of vertices per dimension
        k (int): number of segments
        lambda_ (float): smoothness parameter that represents the cost of assigning two neighboring pixels to different segments

    Returns:
        ndarray: labels for each pixel in the image represented by a (d*d,) array
    """

    lambda_ = lambda_
    probabilities[probabilities == 0] = 1e-10
    grid_edges, grid_edge_weights = get_uniform_smoothness_pw_single_image((d, d))
    # print("mean:", grid_edge_weights.mean(), "std:", grid_edge_weights.std(), "min:", grid_edge_weights.min(), "max:", grid_edge_weights.max(), "sum:", grid_edge_weights.sum() * lambda_)
    unary_vec = -1 * np.log(probabilities.reshape(d * d, k))
    pairwise_pot = (1 - np.eye(k)) * lambda_
    labels = gco.cut_general_graph(grid_edges, 
                                grid_edge_weights, 
                                unary_vec, 
                                pairwise_pot, 
                                algorithm="swap")
    result = labels.reshape(d * d)
    return result

def get_fc_grid_edges(grid_shape):
    n_nodes = grid_shape[0] * grid_shape[1]
    indices = np.arange(n_nodes)
    edges = np.stack(np.meshgrid(indices, indices), -1)
    edges = edges[np.logical_not(np.eye(n_nodes, dtype=bool))].reshape(-1, 2)
    return edges

def get_edge_weights(edges, features, vertex_coords):
    feature_distances = np.zeros((edges.shape[0], 1))
    feature_distances = np.linalg.norm(features[edges[:, 0]] - features[edges[:, 1]], axis=1)
    spatial_distances = np.linalg.norm(vertex_coords[edges[:, 0]] - vertex_coords[edges[:, 1]], axis=1)
    sigma = np.std(feature_distances)
    if sigma == 0:
        sigma = 1
    weights = np.exp((-(feature_distances ** 2) / (2 * sigma))) * (1 / spatial_distances)
    return weights

def graph_cut_with_custom_weights(probabilities, embeddings, d, k, lambda_, cached_components=None):
    """Performs graph cut on a single image based on provided unary potentials in the form of probabilities as well as edge weights generated from embedding similarity

    Args:
        probabilities (ndarray): unary potentials for each pixel in the image represented by a (d*d, n_segments) array
        embeddings (ndarray): embeddings for each pixel in the image represented by a (d*d, embedding_dimensionality) array
        d (int): number of vertices per dimension
        k (int): number of segments
        lambda_ (float): smoothness parameter that represents the cost of assigning two neighboring pixels to different segments

    Returns:
        ndarray: labels for each pixel in the image represented by a (d*d,) array
    """
    if cached_components is None:
        vertex_coords = np.array([[(j,i) for i in range(d)] for j in range(d)]).reshape(d*d, 2)

        # create grid edges and respective (generic) weights
        grid_edges = get_fc_grid_edges((d, d))
        grid_edge_weights = get_edge_weights(grid_edges, embeddings, vertex_coords)

        valid_edges = grid_edge_weights >= 0.2

        if valid_edges.sum() > 0:
            grid_edges = grid_edges[valid_edges]
            grid_edge_weights = grid_edge_weights[valid_edges]

        # create pairwise potential
        mean_edge_weight = np.mean(grid_edge_weights)
        pairwise_pot = (1 - np.eye(k)) * lambda_
    else:
        grid_edges = cached_components["grid_edges"]
        grid_edge_weights = cached_components["grid_edge_weights"]
        pairwise_pot = cached_components["pairwise_pot"]
    
    # Use onehot as unary
    unary_vec = -1 * np.log(probabilities.reshape(d * d, k))
    
    # Perform graph cut
    labels = gco.cut_general_graph(grid_edges, 
                                grid_edge_weights, 
                                unary_vec, 
                                pairwise_pot, 
                                algorithm="swap")

    cache = {
        "grid_edges": grid_edges,
        "grid_edge_weights": grid_edge_weights,
        "pairwise_pot": pairwise_pot
    }
    return labels, cache

def circular_indexing(start, end, step, list_length):
    """Returns a list of 3-tuples (start, end, step) that can be used to index
    a list of length list_length. Start must be less than list_length, but end
    may be greater than list_length. The function treats the list as circular,
    so if end is greater than list_length, the indexing will wrap around to the
    beginning of the list.
    """

    sublast = -1
    while end > 0:
        if sublast == -1:
            substart = start
        elif sublast == list_length:
            substart = 0
        else:
            substart = (sublast + step) % list_length
        subend = min(end, list_length)
        end -= subend
        size = (subend - substart) // step
        sublast = (substart + ((size) * step))
        yield (substart, subend, step)

class PatchDL(torch.utils.data.Dataset):
    def __init__(self, image_tensor, initial_labels, d, batch_size):
        """Dataloader for image patches

        Args:
            image_tensor (tensor): input image represented by a (1, C, H, W) tensor
            initial_labels (tensor): initial labels for each trainining patch represented by a (d^2, n_segments) tensor
            d (_type_): _description_
            batch_size (_type_): _description_
        """
        self.image_tensor = image_tensor
        self.labels = initial_labels
        self.d = d
        self.patch_size = (image_tensor.shape[2] // d, image_tensor.shape[3] // d)
        self.patches = torch.nn.functional.unfold(self.image_tensor, kernel_size=self.patch_size, stride=1, dilation=1, padding=0)
        self.patches = self.patches.reshape(self.image_tensor.shape[1], self.patch_size[0], self.patch_size[1], -1).permute(3, 0, 1, 2).to(device)
        self.train_indices = torch.cat([torch.arange(self.d) * self.patch_size[0] + ((self.image_tensor.shape[2] - (self.patch_size[0] - 1)) * self.patch_size[1] * row) for row in range(d)])
        self.train_indices = torch.stack([self.train_indices, torch.arange(self.d * self.d)], dim=1)
        self.train_patches = self.patches[self.train_indices[:, 0]]
        self.batch_size = int(batch_size * len(self.train_indices))
        self.blank_label = torch.tensor([-1]).to(device)
        self.idx = 0

    def __len__(self):
        return float("inf")
    
    def __next__(self):
        spacing = (self.d**2) // self.batch_size
        slice_gen = circular_indexing(self.idx % self.d**2, (self.idx % self.d**2) + spacing * self.batch_size, spacing, self.d**2)
        slices = list(slice_gen)
        patches = torch.cat([self.train_patches[start:end:step] for start, end, step in slices], dim=0)
        labels = torch.cat([self.labels[start:end:step] for start, end, step in slices], dim=0)
        indices = torch.cat([torch.arange(start,end,step) for start, end, step in slices])
        self.idx += 1
        return patches, labels, indices, slices
        
    def get_relabel_set(self):
        """Returns all patches in the "training set" along with their corresponding labels and indices

        Returns:
            tensor: training patches represented by a (d^2, C, H, W) tensor
            tensor: training labels represented by a (d^2, n_segments) tensor
            tensor: indices of training patches represented by a (d^2,) tensor
        """
        patches = self.patches[self.train_indices[:, 0]]
        labels = self.labels
        indices = self.train_indices[:, 1]
        return patches, labels, indices
    
    def get_inference_set(self):
        """Returns all patches in the image

        Returns:
            tensor: all patches represented by a (d^2, C, H, W) tensor
        """
        patches = self.patches
        return patches
    
    def get_inference_batches(self, batch_size):
        dl = torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=False)
        return dl

    
def get_DINO_embeddings(img_, d, dimensions=384):
    """Gets DINO patch embeddings for an image decomposed into the desired number of components using PCA with a polynomial kernel

    Args:
        img_ (tensor): input image represented by a (1, C, H, W) tensor
        d (int): number of patches across
        dimensions (int, optional): number of dimensions to reduce to. Defaults to 384.

    Returns:
        tensor: DINO embeddings for each patch represented by a (d^2, dimensions) tensor
    """
    # resize image so that largest dimension divided by 14 is d
    img = torch.nn.functional.interpolate(img_, (14 * d, 14 * d), mode="bilinear")
    input_size = img.shape[-2:]
    input_size = torch.tensor(input_size)
    dino_size = tuple((torch.ceil(input_size[0] / 14) * 14, torch.ceil(input_size[1] / 14) * 14))
    input_size = tuple(input_size)
    initialize_dino()
    global dinov2
    with torch.no_grad():
        scale_size = max(dino_size)
        crop_size = min(dino_size)
        image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(scale_size), antialias=None),
            torchvision.transforms.CenterCrop(int(crop_size)),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
        img = image_transform(img)
        img = img[:3].to(device)
        embeddings = dinov2.forward_features(img)["x_norm_patchtokens"].squeeze(0)
    dino_size = (int(dino_size[0]), int(dino_size[1]))
    if embeddings.shape[1] > dimensions:
        pca = KernelPCA(n_components=dimensions, kernel="poly")
        embeddings = pca.fit_transform(embeddings.cpu().numpy())
        embeddings = torch.tensor(embeddings)
    embeddings = embeddings.reshape(img.shape[-2] // 14, img.shape[-1] // 14, -1)
    return embeddings, img

def get_color_embeddings(patches, d, dimensions=3):
    patch_avg_colors = patches.mean(dim=(2,3))
    pca = KernelPCA(n_components=dimensions, kernel="poly")
    patch_avg_colors = pca.fit_transform(patch_avg_colors.cpu().numpy())
    patch_avg_colors = patch_avg_colors.reshape(d * d, dimensions)
    patch_avg_colors *= 32
    return patch_avg_colors

    
class GraPLNet(torch.nn.Module):
    def __init__(self, patch_size=(32,32), n_input_channels=3, k=10, n_filters=16, dropout=0.2, bottleneck_dim=4, num_layers=3):
        """FCN architecture which operates on patches rather than the entire image

        Args:
            patch_size (tuple, optional): shape of input patches. Defaults to (32,32).
            k (int, optional): number of output channels (segments). Defaults to 10.
            n_filters (int, optional): number of filters/channels in the middle of the network. Defaults to 16.
            dropout (float, optional): amount of dropout during training. Defaults to 0.2.
        """
        self.num_layers = num_layers
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        super(GraPLNet, self).__init__()
        self.k = k
        self.patch_size = patch_size
        self.n_input_channels = n_input_channels
        self.padding = 0
        padding_compensation = -(2 * (num_layers - 1)) + 4 * self.padding
        self.dropout0 = torch.nn.Dropout((2 * patch_size[0] + 2 * patch_size[1] - 4)/(patch_size[0] * patch_size[1]))
        self.conv1 = torch.nn.Conv2d(self.n_input_channels, n_filters if num_layers > 2 else bottleneck_dim, 3, padding=self.padding)
        self.BN1 = torch.nn.BatchNorm2d(n_filters if num_layers > 2 else bottleneck_dim)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.other_conv = torch.nn.ModuleList()
        self.other_BN = torch.nn.ModuleList()
        self.other_dropout = torch.nn.ModuleList()
        for i in range(num_layers - 3):
            self.other_conv.append(torch.nn.Conv2d(n_filters, n_filters, 3, padding=self.padding))
            self.other_BN.append(torch.nn.BatchNorm2d(n_filters))
            self.other_dropout.append(torch.nn.Dropout(dropout))
        if num_layers > 2:
            self.conv2 = torch.nn.Conv2d(n_filters, bottleneck_dim, 3, padding=self.padding)
            self.BN2 = torch.nn.BatchNorm2d(bottleneck_dim)
            self.dropout2 = torch.nn.Dropout(dropout)
        self.output = torch.nn.Conv2d(bottleneck_dim, k, (patch_size[0] + padding_compensation, patch_size[1] + padding_compensation), 1)
        self.tile_size = patch_size
        self.train_indices = None
        self.use_subset = True
        self.unfold_stride = 1
        self.make_patches = True
        self.patch_cache = None

    def forward(self, x, slices=None, indices=None):
        if slices is not None or indices is not None:
            if self.patch_cache is None:
                patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size, dilation=1, padding=0)
                patches = patches.view(x.shape[1], self.patch_size[0], self.patch_size[1], -1).permute(3, 0, 1, 2)
                self.patch_cache = patches
            else:
                patches = self.patch_cache
            if slices is not None:
                x = torch.cat([patches[start:end:step] for start, end, step in slices])
            else:
                x = patches[indices]
        x = self.dropout0(x)
        x = self.conv1(x)
        # print(x.shape)
        x = self.BN1(x)
        x = torch.tanh(x)
        x = self.dropout1(x)
        for i in range(len(self.other_conv)):
            x = self.other_conv[i](x)
            # print(x.shape)
            x = self.other_BN[i](x)
            x = torch.tanh(x)
            x = self.other_dropout[i](x)
        if self.num_layers > 2:
            x = self.conv2(x)
            # print(x.shape)
            x = self.BN2(x)
            x = torch.tanh(x)
            x = self.dropout2(x)
        x = self.output(x)
        # print(x.shape)
        return x
    
class ContinuityLoss(torch.nn.Module):
    def __init__(self, input_shape=(512,512), continuity_range=1, p=1, diagonal=False):
        super(ContinuityLoss, self).__init__()
        self.input_shape = input_shape
        self.continuity_range = continuity_range
        self.p = p
        self.diagonal = diagonal

    def forward(self, outputs):
        predictions = outputs.reshape(self.input_shape[0], self.input_shape[1], -1)
        predictions = predictions.softmax(dim=-1)
        sum_ = 0
        for i in range(1, self.continuity_range + 1):
            ax = predictions[i:, :, :].clone()
            bx = predictions[:-i, :, :].clone()
            sum_ = torch.norm(ax - bx, p=self.p)
            ay = predictions[:, i:, :]
            by = predictions[:, :-i, :]
            sum_ = sum_ + torch.norm(ay - by, p=self.p)
            if self.diagonal:
                a1 = predictions[i:, i:, :]
                b1 = predictions[:-i, :-i, :]
                sum_ = sum_ + torch.norm(a1 - b1, p=self.p)
                a2 = predictions[i:, :-i, :]
                b2 = predictions[:-i, i:, :]
                sum_ = sum_ + torch.norm(a2 - b2, p=self.p)
                a3 = predictions[:, i:, i:]
                b3 = predictions[:, :-i, :-i]
                sum_ = sum_ + torch.norm(a3 - b3, p=self.p)
                a4 = predictions[:, i:, :-i]
                b4 = predictions[:, :-i, i:]
                sum_ = sum_ + torch.norm(a4 - b4, p=self.p)
        if self.p == 1:
            sum_ = sum_ / (self.input_shape[0] * self.input_shape[1])
        else:
            sum_ = sum_ / np.sqrt(self.input_shape[0] * self.input_shape[1])
        return sum_

class GraPL_Segmentor:
    def __init__(self, d=64, n_filters=16, dropout=0.2,
                lambda_=4, size=(512, 512), lr=0.01,
                iterations=100, subset_size=0.5,
                sigma=5, compactness=0.1, k=10, epochs=40,
                epoch_schedule=epoch_schedule, bottleneck_dim=4,
                max_epochs=40, min_epochs=10, use_embeddings=False,
                use_continuity_loss=False, continuity_range=1, continuity_p=1, continuity_weight=1,
                use_min_loss=False, use_coords=False,
                use_crf=False, use_color_distance_weights=False,
                initialization_method="slic", use_fully_connected=False, num_layers=3, use_collapse_penalty=False,
                seed=None, use_cold_start=False, use_graph_cut=True):
        self.d = d
        self.n_filters = n_filters
        self.dropout = dropout
        self.lambda_ = lambda_
        self.size = size
        self.tile_size = (size[0] // d, size[1] // d)
        self.lr = lr
        self.iterations = iterations
        self.subset_size = subset_size
        self.net = None
        self.losses = []
        self.intermediate_partitions = []
        self.intermediate_probabilities = []
        self.intermediate_graphs = []
        self.intermediate_cross_entropies = []
        self.intermediate_continuities = []
        self.slic_segments = k
        self.sigma = sigma
        self.compactness = compactness
        self.k = k
        self.initial_labels = None
        self.epochs = epochs
        self.epoch_schedule = epoch_schedule
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.initial_segmentation = None
        self.use_embeddings = use_embeddings
        self.embeddings = None
        self.original_image_size = None
        self.use_coords = use_coords
        self.bottleneck_dim = bottleneck_dim
        self.use_continuity_loss = use_continuity_loss
        self.continuity_range = continuity_range
        self.continuity_p = continuity_p
        self.continuity_weight = continuity_weight
        self.use_min_loss = use_min_loss
        self.cached_graph_components = None
        self.num_layers = num_layers
        self.use_color_distance_weights = use_color_distance_weights
        self.initialization_method = initialization_method
        self.use_fully_connected = use_fully_connected
        self.use_collapse_penalty = use_collapse_penalty
        self.use_cold_start = use_cold_start
        self.use_graph_cut = use_graph_cut
        self.intermediate_collapse_penalties = []
        self.prediction = None
        self.training_time = 0
        self.prediction_time = 0
        self.seed = seed
        self.dataloader = None
        if self.seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def fit(self, image):
        """Fits the network to the image

        In this minimal example, the fit method standardizes the image, generates labels using `inital_labels()` (SLIC), and trains the network in a supervised manner, using cross entropy loss.

        Args:
            image (ndarray): input image represented by a (H,W,C) array
        """
        training_start = time.time()
        self.image_tensor = torch.tensor(image, dtype=torch.float32).to(device).permute(2, 0, 1).unsqueeze(0)
        self.original_image_size = image.shape[:2]
        resized_image_width = int(np.ceil(np.max(self.image_tensor.shape[1:]) / self.d) * self.d)
        self.image_tensor = torch.nn.functional.interpolate(self.image_tensor, size=(resized_image_width, resized_image_width))
        self.image = self.image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        image = self.image

        # get embeddings
        if self.use_embeddings:
            self.embeddings, _ = get_DINO_embeddings(self.image_tensor, self.d, dimensions=(self.k))
            self.embeddings = self.embeddings.reshape(-1, self.embeddings.shape[-1])

        self.image_size = self.image_tensor.shape[-2:]
        self.patch_size = (self.image_size[0] // self.d, self.image_size[1] // self.d)

        # standardize image to [-1,1]
        cur_min = self.image_tensor.min()
        cur_max = self.image_tensor.max()
        self.image_tensor = (2 * (self.image_tensor - cur_min)/(cur_max - cur_min)) - 1
        x_grid, y_grid = torch.meshgrid(torch.linspace(-1, 1, self.image_tensor.shape[2]), torch.linspace(-1, 1, self.image_tensor.shape[3]), indexing="ij")
        coord_grid = torch.stack((x_grid, y_grid), dim=0).unsqueeze(0).to(device)
        if self.use_coords:
            self.image_tensor = torch.cat((self.image_tensor, coord_grid), dim=1)

        # create labels using SLIC
        self.initial_labels, self.initial_segmentation = initial_labels(image, self.d, self.k, sigma=self.sigma, compactness=self.compactness, method=self.initialization_method)
        self.initial_labels = self.initial_labels.argmax(dim=1).to(device)
        self.k = self.initial_labels.max() + 1

        # create dataloader
        self.dataloader = PatchDL(self.image_tensor, self.initial_labels, self.d, self.subset_size)
        self.batches = self.dataloader

        if self.use_color_distance_weights:
            patches, _, _ = self.dataloader.get_relabel_set()
            patch_avg_colors = get_color_embeddings(patches, self.d, dimensions=3)

        # Initialize CNN
        self.net = GraPLNet(patch_size=self.patch_size, n_filters=self.n_filters, n_input_channels=self.image_tensor.shape[1], dropout=self.dropout, k=self.k, bottleneck_dim=self.bottleneck_dim, num_layers=self.num_layers).to(device)
        self.net.train()
        self.previous_params = copy.deepcopy(self.net.state_dict())
        self.initial_params = copy.deepcopy(self.net.state_dict())

        # Initialize optimizer and loss function
        cross_entropy = torch.nn.CrossEntropyLoss()
        continuity_loss = ContinuityLoss(input_shape=(self.d,self.d), continuity_range=self.continuity_range, p=self.continuity_p, diagonal=False)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        most_recent_outputs_full = torch.nn.functional.one_hot(self.initial_labels, num_classes=self.k).float().to(device)

        # Train CNN
        for iteration in range(self.iterations):
            n_epochs = self.epoch_schedule(iteration, max_epochs=self.max_epochs, min_epochs=self.min_epochs, n_iter=self.iterations)
            for epoch in range(n_epochs):
                _, labels, indices, slices = next(self.batches)
                optimizer.zero_grad()
                outputs = self.net(self.image_tensor, slices=slices).squeeze(-1).squeeze(-1)
                fill_line = 0
                for start, end, step in slices:
                    # length = (end - start) // step
                    length = most_recent_outputs_full[start:end:step].shape[0]
                    most_recent_outputs_full[start:end:step] = outputs[fill_line:fill_line + length]
                    fill_line += length

                # most_recent_outputs_full[indices] = outputs
                ce = cross_entropy(outputs, labels)
                # self.intermediate_cross_entropies.append(ce.item())
                loss = ce
                if self.use_continuity_loss:
                    continuity = continuity_loss(most_recent_outputs_full) * self.continuity_weight
                    loss = loss + continuity
                    # self.intermediate_continuities.append(continuity.item())
                if self.use_collapse_penalty:
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    entropy = - torch.sum(probs * torch.log(probs), dim=1)
                    collapse_penalty = torch.mean(entropy)
                    loss = loss + collapse_penalty
                    # self.intermediate_collapse_penalties.append(collapse_penalty.item())
                loss.backward()
                most_recent_outputs_full = most_recent_outputs_full.detach()
                optimizer.step()
                if iteration == 0 and loss.item() < 1:
                    # print("stopping first iteration early on epoch", epoch)
                    break
            
            previous_params = copy.deepcopy(self.net.state_dict())

            # Skip graph step on last iteration
            if iteration != self.iterations - 1:
                # Perform graph step
                self.net.eval()
                _, _, indices = self.dataloader.get_relabel_set()
                outputs = self.net(self.image_tensor, indices=indices).squeeze(-1).squeeze(-1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()
                if not self.use_graph_cut:
                    partition = probabilities
                elif self.use_embeddings:
                    partition, self.cached_graph_components = graph_cut_with_custom_weights(probabilities, self.embeddings, self.d, self.k, self.lambda_, cached_components=self.cached_graph_components)
                    partition = torch.tensor(partition, dtype=torch.int64)
                    partition = torch.nn.functional.one_hot(partition, self.k)
                elif self.use_color_distance_weights:
                    partition, self.cached_graph_components = graph_cut_with_custom_weights(probabilities, patch_avg_colors, self.d, self.k, self.lambda_, cached_components=self.cached_graph_components)
                    partition = torch.tensor(partition, dtype=torch.int64)
                    partition = torch.nn.functional.one_hot(partition, self.k)
                elif self.use_fully_connected:
                    modified_lambda_ = self.lambda_ / self.d
                    partition, self.cached_graph_components = graph_cut_with_custom_weights(probabilities, torch.ones(self.d ** 2, 1), self.d, self.k, modified_lambda_, cached_components=self.cached_graph_components)
                    partition = torch.tensor(partition, dtype=torch.int64)
                    partition = torch.nn.functional.one_hot(partition, self.k)
                else:
                    partition = torch.tensor(graph_cut(probabilities, self.d, self.k, self.lambda_), dtype=torch.int64)
                    partition = torch.nn.functional.one_hot(partition, self.k)
                self.dataloader.labels = torch.Tensor(partition).to(device).argmax(dim=1).type(torch.long)
                if self.use_cold_start:
                    self.net = GraPLNet(patch_size=self.patch_size, n_filters=self.n_filters, n_input_channels=self.image_tensor.shape[1], dropout=self.dropout, k=self.k, bottleneck_dim=self.bottleneck_dim, num_layers=self.num_layers).to(device)
                    self.net.load_state_dict(self.initial_params)
                    optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
                self.net.train()
        self.training_time = time.time() - training_start
    
    def predict(self):
        prediction_start = time.time()
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(self.image_tensor).detach()
            outputs = torch.nn.functional.interpolate(outputs, self.image_tensor.shape[2:])
            seg = outputs.argmax(1).squeeze(0).cpu().numpy()
            if self.use_coords:
                img = self.image
                img = img - img.min()
                img = img / img.max()
                img = img * 255
                img = img.astype(np.uint8)
                g = skimage.graph.rag_mean_color(img, seg, mode='similarity', sigma=255.0)
                seg = skimage.graph.cut_normalized(seg, g, thresh=0.1)
        seg = torch.nn.functional.interpolate(torch.tensor(seg).unsqueeze(0).unsqueeze(0).float(), self.original_image_size).squeeze(0).squeeze(0).numpy()
        self.prediction = seg
        self.prediction_time = time.time() - prediction_start
        return seg
    
    def predict_patchwise(self):
        self.net.eval()
        with torch.no_grad():
            patches, _, indices = self.dataloader.get_relabel_set()
            outputs = self.net(patches, indices=indices).detach()
            outputs = torch.softmax(outputs, dim=1)
            seg = outputs.reshape(self.d,self.d,self.k)
        return seg
    
    def save_seg(self, path):
        # colormap segmentation with viridis
        norm = matplotlib.colors.Normalize(vmin=self.prediction.min(), vmax=self.prediction.max())
        m = colormap.ScalarMappable(norm=norm, cmap=colormap.viridis)
        color_seg = (m.to_rgba(self.prediction)[:,:,:3] * 255).astype(np.uint8)
        img = Image.fromarray(color_seg)
        # add metadata
        metadata = PngInfo()
        metadata.add_text("training_time", str(self.training_time))
        metadata.add_text("prediction_time", str(self.prediction_time))
        metadata.add_text("total_time", str(self.training_time + self.prediction_time))
        metadata.add_text("iterations", str(self.iterations))
        metadata.add_text("lambda_", str(self.lambda_))
        metadata.add_text("k", str(self.k.item()))
        metadata.add_text("d", str(self.d))
        metadata.add_text("use_coords", str(self.use_coords))
        metadata.add_text("use_embeddings", str(self.use_embeddings))
        metadata.add_text("use_color_distance_weights", str(self.use_color_distance_weights))
        metadata.add_text("use_fully_connected", str(self.use_fully_connected))
        metadata.add_text("n_filters", str(self.n_filters))
        metadata.add_text("dropout", str(self.dropout))
        metadata.add_text("lr", str(self.lr))
        metadata.add_text("max_epochs", str(self.epochs))
        metadata.add_text("min_epochs", str(self.min_epochs))
        metadata.add_text("use_collapse_penalty", str(self.use_collapse_penalty))
        metadata.add_text("bottleneck_dim", str(self.bottleneck_dim))
        metadata.add_text("use_continuity_loss", str(self.use_continuity_loss))
        metadata.add_text("continuity_range", str(self.continuity_range))
        metadata.add_text("continuity_p", str(self.continuity_p))
        metadata.add_text("continuity_weight", str(self.continuity_weight))
        metadata.add_text("use_min_loss", str(self.use_min_loss))
        metadata.add_text("initialization_method", str(self.initialization_method))
        metadata.add_text("seed", str(self.seed))
        img.save(path, pnginfo=metadata)
    
def side_by_side(images, titles=None, height=3):
    """Display images side by side.

    Args:
        images (list): list of images to display
        titles (list): list of titles for each image
        height (int): height of each image
    """
    fig, axes = plt.subplots(1, len(images), figsize=(height*len(images), height))
    if len(images) == 1:
        axes = [axes]
    for i, image in enumerate(images):
        axes[i].imshow(image)
        axes[i].axis("off")
        if titles is not None:
            axes[i].set_title(titles[i])
    plt.show()

def view_multichannel(img):
    """Displays an image with >3 channels by dividing the channels into groups of <3 and displaying each group in a separate subplot.

    Args:
        img (np.ndarray): image to display
    """
    n_channels = img.shape[-1]
    n_cols = int(np.ceil(n_channels / 3)) + 1
    fig, axarr = plt.subplots(ncols=n_cols, figsize=(n_cols * 2, 2))
    for i in range(n_cols-1):
        channels = img[:, :, i*3:min(n_channels,(i+1)*3)]
        if channels.shape[-1] == 2:
            channels = np.concatenate([channels, np.zeros((*channels.shape[:-1], 1))], axis=-1) 
        channels = axarr[i].imshow(channels, vmin=0, vmax=1)
        axarr[i].set_title(f"channels {i*3} - {min(n_channels,(i+1)*3)}")
    axarr[-1].imshow(img.argmax(-1))
    axarr[-1].set_title("argmax")
    plt.show()


def hyper_comparison(paramsets, n_images=10, trials=10):
    image_paths = glob.glob("datasets/BSDS500/BSDS500/data/images/test/*.jpg")[0:n_images]
    scores = {}
    for name in paramsets:
        paramset = paramsets[name]
        set_scores = {}
        for image_path in image_paths:
            id = image_path.split("/")[-1].split(".")[0]
            image = np.array(Image.open(image_path).resize((512, 512)))[:,:,:3]
            for trial in range(trials):
                segmentor = GraPL_Segmentor(**paramset)
                start_time = time.time()
                segmentor.fit(image)
                seg = segmentor.predict()
                duration = time.time() - start_time
                trial_scores = bsds_score(id, seg)
                trial_scores = {metric: np.mean(trial_scores[metric]) for metric in trial_scores}
                trial_scores["time"] = duration
                for metric in trial_scores:
                    if metric not in set_scores:
                        set_scores[metric] = []
                    set_scores[metric].append(trial_scores[metric])
        set_scores = {metric: np.mean(set_scores[metric]) for metric in set_scores}
        scores[name] = set_scores
    return scores

hyperparameter_profiles = {
    'best_miou': {
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
        'use_coords': False,
        'use_embeddings': False,
        'use_color_distance_weights': True,
        'seed': 0,
        'use_continuity_loss': True,
        'continuity_range': 1,
        'continuity_weight': 3,
        'use_min_loss': True
    }
}

def segment(image, save_path=None, **params):
    # if params not provided, use default
    if len(params) == 0:
        params = hyperparameter_profiles['best_miou']
    elif "profile" in params:
        params = hyperparameter_profiles[params["profile"]]
    segmentor = GraPL_Segmentor(**params)
    segmentor.fit(image)
    seg = segmentor.predict()
    if save_path is not None:
        segmentor.save_seg(save_path)
    torch.cuda.empty_cache()
    return seg

def segment_bsds(results_dir=None, resume=False, progress_bar=None, **hyperparams):
    if results_dir is None:
        results_dir = f'results/BSDS_{int(time.time())}'
    os.makedirs(results_dir, exist_ok=True)
    image_paths = glob.glob('datasets/BSDS500/BSDS500/data/images/test/*.jpg')[:]
    paramset_scores = {}
    if progress_bar is None:
        progress_bar = tqdm.tqdm(total=len(image_paths))
    for image_path in image_paths:
        id = image_path.split('/')[-1].split('.')[0]
        image = plt.imread(image_path)[:,:,:3]
        image_scores = {}
        save_path = f'{results_dir}/{id}.png'
        if resume and os.path.exists(save_path):
            image_scores = bsds_score(id, save_path)
            paramset_scores[id] = image_scores
            progress_bar.update(1)
            continue
        seg = segment(image, save_path=save_path, **hyperparams)
        image_scores = bsds_score(id, save_path)
        paramset_scores[id] = image_scores
        progress_bar.update(1)
    with open(f'{results_dir}/scores.json', 'w') as fp:
        results = {"hyperparams": hyperparams, "scores": paramset_scores}
        json.dump(results, fp)
    return paramset_scores

def segment_voc(results_dir=None, resume=False, progress_bar=None, debug_num=-1, **hyperparams):
    if results_dir is None:
        results_dir = f'results/BSDS_{int(time.time())}'
    os.makedirs(results_dir, exist_ok=True)
    with open("datasets/PascalVOC2012/VOC2012/ImageSets/Segmentation/val.txt", "r") as f:
        val_image_ids = f.read().split("\n")
        val_image_ids = [id for id in val_image_ids if id != ""]
    image_paths = [f"datasets/PascalVOC2012/VOC2012/JPEGImages/{id}.jpg" for id in val_image_ids]
    paramset_scores = {}
    if progress_bar is None:
        progress_bar = tqdm.tqdm(total=len(image_paths))
    if debug_num > 0:
        image_paths = image_paths[:debug_num]
    for image_path in image_paths:
        id = image_path.split('/')[-1].split('.')[0]
        image = plt.imread(image_path)[:,:,:3]
        image_scores = {}
        save_path = f'{results_dir}/{id}.png'
        if resume and os.path.exists(save_path):
            image_scores = voc_score(id, save_path)
            paramset_scores[id] = image_scores
            progress_bar.update(1)
            continue
        seg = segment(image, save_path=save_path, **hyperparams)
        image_scores = voc_score(id, save_path)
        paramset_scores[id] = image_scores
        progress_bar.update(1)
    with open(f'{results_dir}/scores.json', 'w') as fp:
        results = {"hyperparams": hyperparams, "scores": paramset_scores}
        json.dump(results, fp)
    return paramset_scores