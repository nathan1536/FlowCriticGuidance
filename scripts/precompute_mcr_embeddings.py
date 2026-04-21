"""
Precompute MCR (ResNet50) image embeddings for all frames in a zarr replay buffer.
Saves `img_embedding` (N, 2048) and `next_img_embedding` (N, 2048) back into the zarr.

Run once before training:
    python scripts/precompute_mcr_embeddings.py \
        --zarr-path runs/sb3_adroit_sac/models_84/door/replay_buffer_t1000000_84x.zarr \
        --ckpt-path robots-pretrain-robots/mcr/ckpts/mcr_resnet50.pth \
        --batch-size 512 \
        --device cuda
"""
import argparse
import sys
import pathlib
import numpy as np
import torch
import torchvision.transforms as T
import zarr
import tqdm
from PIL import Image

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
MCR_DIR = str(pathlib.Path(ROOT_DIR) / 'robots-pretrain-robots')
sys.path.insert(0, MCR_DIR)

# MCR preprocessing — matches example.py exactly
MCR_TRANSFORMS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),   # divides by 255, outputs (C, H, W)
])


def encode_batch(model, imgs_np: np.ndarray, device: torch.device) -> np.ndarray:
    """
    imgs_np : (B, H, W, C) uint8
    returns  : (B, 2048) float32
    """
    tensors = [MCR_TRANSFORMS(Image.fromarray(img)) for img in imgs_np]
    x = torch.stack(tensors).to(device)   # (B, 3, 224, 224)
    with torch.no_grad():
        emb = model(x)                     # (B, 2048)
    return emb.cpu().numpy().astype(np.float32)


def encode_array(model, img_zarr, batch_size: int, device: torch.device,
                 desc: str) -> np.ndarray:
    """Encode entire zarr image array in batches, return (N, 2048) numpy array."""
    N = img_zarr.shape[0]
    out = np.zeros((N, 2048), dtype=np.float32)
    for start in tqdm.trange(0, N, batch_size, desc=desc):
        end = min(start + batch_size, N)
        batch = img_zarr[start:end]        # (B, H, W, C) uint8
        out[start:end] = encode_batch(model, batch, device)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr-path', required=True,
                        help='Path to zarr replay buffer (e.g. runs/sb3_metaworld_sac/models_84/bin-picking/replay_buffer_t6200000.zarr)')
    parser.add_argument('--ckpt-path', required=True,
                        help='Path to MCR checkpoint (e.g. ../robots-pretrain-robots/mcr_resnet50.pth)')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing img_embedding / next_img_embedding arrays')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load MCR
    from mcr import load_mcr
    print(f'Loading MCR from {args.ckpt_path}')
    mcr = load_mcr(ckpt_path=args.ckpt_path)
    mcr.to(device).eval()
    for p in mcr.parameters():
        p.requires_grad = False

    # Open zarr in append mode
    z = zarr.open(args.zarr_path, mode='a')
    img_zarr      = z['data']['img']       # (N, H, W, C) uint8
    # next_img_zarr = z['data']['next_img']  # (N, H, W, C) uint8
    N = img_zarr.shape[0]
    print(f'Zarr has {N} frames, image shape {img_zarr.shape[1:]}')

    # img_embedding
    if 'img_embedding' in z['data'] and not args.overwrite:
        print('img_embedding already exists — skipping (use --overwrite to recompute)')
    else:
        print('Encoding img...')
        emb = encode_array(mcr, img_zarr, args.batch_size, device, desc='img')
        if 'img_embedding' in z['data']:
            del z['data']['img_embedding']
        z['data'].create_dataset('img_embedding', data=emb,
                                 chunks=(100, 2048), dtype='float32')
        print(f'  Saved img_embedding: {z["data"]["img_embedding"].shape}')

    # # next_img_embedding
    # if 'next_img_embedding' in z['data'] and not args.overwrite:
    #     print('next_img_embedding already exists — skipping (use --overwrite to recompute)')
    # else:
    #     print('Encoding next_img...')
    #     next_emb = encode_array(mcr, next_img_zarr, args.batch_size, device, desc='next_img')
    #     if 'next_img_embedding' in z['data']:
    #         del z['data']['next_img_embedding']
    #     z['data'].create_dataset('next_img_embedding', data=next_emb,
    #                              chunks=(100, 2048), dtype='float32')
    #     print(f'  Saved next_img_embedding: {z["data"]["next_img_embedding"].shape}')

    print(z.tree())


if __name__ == '__main__':
    main()
