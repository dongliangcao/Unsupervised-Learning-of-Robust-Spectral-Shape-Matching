# running on a single shape pair
import os
import scipy.io as sio
import torch
import torch.nn.functional as F

from networks.diffusion_network import DiffusionNet
from networks.permutation_network import Similarity

from utils.shape_util import read_shape
from utils.texture_util import write_obj_pair
from utils.geometry_util import compute_operators
from utils.fmap_util import nn_query, fmap2pointmap
from utils.tensor_util import to_numpy


def to_tensor(vert_np, face_np, device):
    vert = torch.from_numpy(vert_np).to(device=device, dtype=torch.float32)
    face = torch.from_numpy(face_np).to(device=device, dtype=torch.long)

    return vert, face


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

    feat_x = feature_extractor(vert_x.unsqueeze(0), face_x.unsqueeze(0))
    feat_y = feature_extractor(vert_y.unsqueeze(0), face_y.unsqueeze(0))
    # normalize features
    feat_x = F.normalize(feat_x, dim=-1, p=2)
    feat_y = F.normalize(feat_y, dim=-1, p=2)

    if non_isometric:
        # nearest neighbour query
        p2p = nn_query(feat_x, feat_y).squeeze()

        # compute Pyx from functional map
        Cxy = evecs_trans_y @ evecs_x[p2p]
        Pyx = evecs_y @ Cxy @ evecs_trans_x
    else:
        # compute Pyx
        similarity = torch.bmm(feat_y, feat_x.transpose(1, 2))
        Pyx = permutation(similarity).squeeze(0)

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
        # save functional map and point-wise correspondences
        save_dict = {'Cxy': Cxy, 'p2p': p2p + 1}  # plus one for MATLAB
        sio.savemat(os.path.join(result_path, f'{name_x}-{name_y}.mat'), save_dict)

    print(f'Finished, see the results under {result_path}')
