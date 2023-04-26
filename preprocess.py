import os
import scipy.io as sio
import numpy as np
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch

from utils.geometry_util import laplacian_decomposition, get_operators
from utils.shape_util import read_shape, compute_geodesic_distmat, write_off


if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser('Preprocess .off files')
    parser.add_argument('--data_root', required=True, help='data root contains /off sub-folder.')
    parser.add_argument('--n_eig', type=int, default=200, help='number of eigenvectors/values to compute.')
    parser.add_argument('--no_eig', action='store_true', help='no laplacian eigen-decomposition')
    parser.add_argument('--no_dist', action='store_true', help='no geodesic matrix.')
    parser.add_argument('--no_normalize', action='store_true', help='no normalization of face area.')
    args = parser.parse_args()

    # sanity check
    data_root = args.data_root
    n_eig = args.n_eig
    no_eig = args.no_eig
    no_dist = args.no_dist
    no_normalize = args.no_normalize
    assert n_eig > 0, f'Invalid n_eig: {n_eig}'
    assert os.path.isdir(data_root), f'Invalid data root: {data_root}'

    if not no_eig:
        spectral_dir = os.path.join(data_root, 'diffusion')
        os.makedirs(spectral_dir, exist_ok=True)

    if not no_dist:
        dist_dir = os.path.join(data_root, 'dist')
        os.makedirs(dist_dir, exist_ok=True)

    # read .off files
    off_files = sorted(glob(os.path.join(data_root, 'off', '*.off')))
    assert len(off_files) != 0

    for off_file in tqdm(off_files):
        verts, faces = read_shape(off_file)
        filename = os.path.basename(off_file)

        if not no_normalize:
            # center shape
            verts -= np.mean(verts, axis=0)

            # normalize verts by sqrt face area
            old_sqrt_area = laplacian_decomposition(verts=verts, faces=faces, k=1)[-1]
            print(f'Old face sqrt area: {old_sqrt_area:.3f}')
            verts /= old_sqrt_area

            # save new verts and faces
            write_off(off_file, verts, faces)

        if not no_eig:
            # recompute laplacian decomposition
            get_operators(torch.from_numpy(verts).float(), torch.from_numpy(faces).long(),
                          k=n_eig, cache_dir=spectral_dir)

        if not no_dist:
            # compute distance matrix
            dist_mat = compute_geodesic_distmat(verts, faces)
            # save results
            sio.savemat(os.path.join(dist_dir, filename.replace('.off', '.mat')), {'dist': dist_mat})
