# running on a single shape pair
import os
import scipy.io as sio
from tqdm.auto import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from networks.diffusion_network import DiffusionNet
from networks.permutation_network import Similarity
from networks.fmap_network import RegularizedFMNet

from utils.shape_util import read_shape
from utils.texture_util import write_obj_pair
from utils.geometry_util import compute_operators
from utils.fmap_util import nn_query, fmap2pointmap
from utils.tensor_util import to_numpy

from losses.fmap_loss import SURFMNetLoss, PartialFmapsLoss, SquaredFrobeniusLoss
from losses.dirichlet_loss import DirichletLoss


def to_tensor(vert_np, face_np, device):
    vert = torch.from_numpy(vert_np).to(device=device, dtype=torch.float32)
    face = torch.from_numpy(face_np).to(device=device, dtype=torch.long)

    return vert, face


def compute_features(vert_x, face_x, vert_y, face_y, feature_extractor, normalize=False):
    feat_x = feature_extractor(vert_x.unsqueeze(0), face_x.unsqueeze(0))
    feat_y = feature_extractor(vert_y.unsqueeze(0), face_y.unsqueeze(0))
    # normalize features
    if normalize:
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)

    return feat_x, feat_y


def compute_permutation_matrix(feat_x, feat_y, permutation, bidirectional=False, normalize=True):
    # normalize features
    if normalize:
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)
    similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

    Pxy = permutation(similarity)

    if bidirectional:
        Pyx = permutation(similarity.transpose(1, 2))
        return Pxy, Pyx
    else:
        return Pxy


def update_network(loss_metrics, feature_extractor, optimizer):
    # compute total loss
    loss = 0.0
    for k, v in loss_metrics.items():
        if k != 'l_total':
            loss += v
    optimizer.zero_grad()
    # backward pass
    loss.backward()
    # clip gradient for stability
    torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), 1.0)
    # update weight
    optimizer.step()

    return loss


if __name__ == '__main__':
    # read shape pair
    filename1 = '/data/caodongliang/FAUST_r/off/tr_reg_080.off'
    filename2 = '/data/caodongliang/FAUST_r/off/tr_reg_081.off'

    vert_np_x, face_np_x = read_shape(filename1)
    vert_np_y, face_np_y = read_shape(filename2)

    # convert numpy to tensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vert_x, face_x = to_tensor(vert_np_x, face_np_x, device)
    vert_y, face_y = to_tensor(vert_np_y, face_np_y, device)

    # compute Laplacian
    _, mass_x, Lx, evals_x, evecs_x, _, _ = compute_operators(vert_x, face_x, k=200)
    _, mass_y, Ly, evals_y, evecs_y, _, _ = compute_operators(vert_y, face_y, k=200)
    evecs_trans_x = evecs_x.T * mass_x[None]
    evecs_trans_y = evecs_y.T * mass_y[None]

    # load pretrained network
    network_path = 'checkpoints/faust.pth'
    input_type = 'wks'  # 'xyz'
    in_channels = 128 if input_type == 'wks' else 3  # 'xyz'
    feature_extractor = DiffusionNet(in_channels=in_channels, out_channels=256, input_type=input_type).to(device)
    feature_extractor.load_state_dict(torch.load(network_path)['networks']['feature_extractor'], strict=True)
    permutation = Similarity(tau=0.07, hard=True).to(device)

    # non-isometric or not
    non_isometric = False  # True

    # refine or not
    num_refine = 0  # 15
    # partial or not
    partial = False
    if num_refine > 0:
        feature_extractor.train()
        fmap_net = RegularizedFMNet(bidirectional=True)
        optimizer = optim.Adam(feature_extractor.parameters(), lr=1e-3)
        fmap_loss = SURFMNetLoss(w_bij=1.0, w_orth=1.0, w_lap=0.0) if not partial else PartialFmapsLoss(w_bij=1.0, w_orth=1.0)
        align_loss = SquaredFrobeniusLoss(loss_weight=1.0)
        if non_isometric:
            w_dirichlet = 5.0
        else:
            if partial:
                w_dirichlet = 1.0
            else:
                w_dirichlet = 0.0
        dirichlet_loss = DirichletLoss(loss_weight=w_dirichlet)
        print('Test-time adaptation')
        pbar = tqdm(range(num_refine))
        for _ in pbar:
            feat_x, feat_y = compute_features(vert_x, face_x, vert_y, face_y, feature_extractor)
            Cxy, Cyx = fmap_net(feat_x, feat_y, evals_x.unsqueeze(0), evals_y.unsqueeze(0),
                                evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0))
            Pxy, Pyx = compute_permutation_matrix(feat_x, feat_y, permutation, bidirectional=True)

            # compute functional map regularisation loss
            loss_metrics = fmap_loss(Cxy, Cyx, evals_x.unsqueeze(0), evals_y.unsqueeze(0))
            # compute C
            Cxy_est = torch.bmm(evecs_trans_y.unsqueeze(0), torch.bmm(Pyx, evecs_x.unsqueeze(0)))

            # compute couple loss
            loss_metrics['l_align'] = align_loss(Cxy, Cxy_est)
            if not partial:
                Cyx_est = torch.bmm(evecs_trans_x.unsqueeze(0), torch.bmm(Pxy, evecs_y.unsqueeze(0)))
                loss_metrics['l_align'] += align_loss(Cyx, Cyx_est)

            # compute dirichlet energy
            if non_isometric:
                loss_metrics['l_d'] = (dirichlet_loss(torch.bmm(Pxy, vert_y.unsqueeze(0)), Lx.to_dense().unsqueeze(0)) +
                                       dirichlet_loss(torch.bmm(Pyx, vert_x.unsqueeze(0)), Ly.to_dense().unsqueeze(0)))

            loss = update_network(loss_metrics, feature_extractor, optimizer)
            pbar.set_description(f'Total loss: {loss:.4f}')

    feature_extractor.eval()
    with torch.no_grad():
        feat_x, feat_y = compute_features(vert_x, face_x, vert_y, face_y, feature_extractor)

    if non_isometric:
        # nearest neighbour query
        p2p = nn_query(feat_x, feat_y).squeeze()

        # compute Pyx from functional map
        Cxy = evecs_trans_y @ evecs_x[p2p]
        Pyx = evecs_y @ Cxy @ evecs_trans_x
    else:
        # compute Pyx
        Pyx = compute_permutation_matrix(feat_y, feat_x, permutation, bidirectional=False).squeeze(0)

        # compute Cxy
        Cxy = evecs_trans_y @ (Pyx @ evecs_x)

        # convert functional map to point-to-point map
        p2p = fmap2pointmap(Cxy, evecs_x, evecs_y)

        # compute Pyx from functional map
        Pyx = evecs_y @ Cxy @ evecs_trans_x

    # save texture transfer result
    Pyx = to_numpy(Pyx)
    result_path = 'results'
    name_x = os.path.splitext(os.path.basename(filename1))[0]
    name_y = os.path.splitext(os.path.basename(filename2))[0]
    file_x = os.path.join(result_path, f'{name_x}.obj')
    file_y = os.path.join(result_path, f'{name_x}-{name_y}.obj')
    write_obj_pair(file_x, file_y, vert_np_x, face_np_x, vert_np_y, face_np_y, Pyx, 'figures/texture.png')

    # save results for MATLAB
    save = False  # True
    if save:
        Cxy, p2p = to_numpy(Cxy), to_numpy(p2p)
        # save functional map and point-wise correspondences
        save_dict = {'Cxy': Cxy, 'p2p': p2p + 1}  # plus one for MATLAB
        sio.savemat(os.path.join(result_path, f'{name_x}-{name_y}.mat'), save_dict)

    print(f'Finished, see the results under {result_path}')
