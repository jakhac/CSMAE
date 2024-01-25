from omegaconf import OmegaConf

import torch
import argparse

import faiss
import faiss.contrib.torch_utils
import numpy as np

import pprint
import itertools
import tqdm
import glob
import os
from torch.utils.data import DataLoader

from src.augmentations import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
)

from src.bigearthnet_dataset.BEN_DataModule_LMDB_Encoder import BENDataSet
from src.utils import Messages
from src.csmae_backbone import vit_tiny, vit_small, vit_base, vit_large


def get_features_lists(backbone, data_loader, dev):
    """Feed all batches from data_loader to backbone and return accumulated features in dictionary with keys "s1" and "s2".

    Args:
        backbone (torch.nn): CSMAE backbone
        data_loader (torch.utils.data.Dataloader): Dataloader
        dev (str): device

    Returns:
        Dict[]: Dictionary with keys "s1" and "s2".
    """
    
    target_list = []
    feat_list_s1 = []
    feat_list_s2 = []

    for _, X, y in tqdm.tqdm(data_loader):
        X, y = X[0].to(dev), y
        out = backbone(X)
        assert isinstance(out, dict)

        features_s1 = out['s1']
        features_s2 = out['s2']

        feat_list_s1.append(features_s1.cpu().detach().numpy())
        feat_list_s2.append(features_s2.cpu().detach().numpy())
        
        target_list.append(y.numpy())

    feat_list_s1 = np.concatenate(feat_list_s1)
    feat_list_s2 = np.concatenate(feat_list_s2)
    target_list = np.concatenate(target_list)

    return feat_list_s1, feat_list_s2, target_list


def main(model_id, device):
    """Performs image retrieval for all uni-/cross-modal cases.

    Args:
        model_id (str): 8-character-id of model name to be evaluated. See under ./trained_models
        device (int): GPU device number
    """

    # initialize paths, devices, aux variables
    dev = f"cuda:{device}"
    torch.cuda.set_device(dev)

    path_to_model = f"./trained_models/{model_id}"
    cfg = OmegaConf.load(f"{path_to_model}/args.yaml")
    OmegaConf.set_struct(cfg, False)
    ckpts = glob.glob(f'{path_to_model}/*.ckpt')

    ckpt_path = ckpts[0]
    assert len(ckpts) == 1
    assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")
    model_config_path = os.path.dirname(ckpt_path) + "/args.yaml"
    model_cfg = OmegaConf.load(model_config_path)

    # overwrite backbone config to match the one used for pre-training
    cfg.backbone = model_cfg.backbone
    
    # initialize backbone
    size_2_vit = {
        'vit_tiny': vit_tiny,
        'vit_small': vit_small,
        'vit_base': vit_base,
        'vit_large': vit_large,
    }
    backbone = size_2_vit[cfg.backbone.name](**cfg.backbone.kwargs)

    # load ckpt weights into backbone
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    ret = backbone.load_state_dict(state, strict=True)
    Messages.hint(f"Loaded checkpoint ({ret})")

    # Build data augmentation pipeline, disabling all non-normalizing transforms
    pipelines = []
    for aug_cfg in cfg.augmentations:
        aug_cfg.rrc.enabled = False
        aug_cfg.horizontal_flip.prob = .0
        aug_cfg.vertical_flip.prob = .0
        pipelines.append(
            NCropAugmentation(
                build_transform_pipeline(cfg.data.dataset, aug_cfg, cfg), aug_cfg.num_crops
            )
        )
    transform = FullTransformPipeline(pipelines)

    # Prepare datasets and create dataloaders
    val_dataset = BENDataSet(
        transform=transform,
        root_dir=cfg.data.root_dir,
        split_dir=cfg.data.split_dir,
        split="val",
        max_img_idx=cfg.data.get('max_img_idx', None),
        img_size=(cfg.data.num_bands, cfg.data.img_size, cfg.data.img_size),
    )

    test_dataset = BENDataSet(
        transform=transform,
        root_dir=cfg.data.root_dir,
        split_dir=cfg.data.split_dir,
        split="test",
        max_img_idx=cfg.data.get('max_img_idx', None),
        img_size=(cfg.data.num_bands, cfg.data.img_size, cfg.data.img_size),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.optimizer.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.optimizer.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    backbone.eval()
    backbone.to(dev)

    # query features from train set / archive features from test set
    with torch.no_grad():
        query_s1, query_s2, query_labels = get_features_lists(backbone, val_loader, dev)
        archive_s1, archive_s2, archive_labels = get_features_lists(backbone, test_loader, dev)


    retrieval_dict = {
        'query_s1': query_s1,
        'query_s2': query_s2,
        'archive_s1': archive_s1,
        'archive_s2': archive_s2,
    }

    embed_dim = archive_s1.shape[1]
    print("Archive:", archive_s1.shape, "Labels:", archive_labels.shape)

    k = 10
    index_func = faiss.IndexFlatL2

    faiss.normalize_L2(query_s1)
    faiss.normalize_L2(query_s2)
    faiss.normalize_L2(archive_s1)
    faiss.normalize_L2(archive_s2)

    results = {}
    modalities = ['s1', 's2']
    for from_modality, to_modality in list(itertools.product(modalities, modalities)):

        print(f"Compute IR results for {from_modality}->{to_modality}")

        index = index_func(embed_dim)
        assert index.is_trained, "This index requires training before use."
        index.add(retrieval_dict['archive_' + to_modality])

        # Retrieve k-closest ids
        _, topk_ids_list = index.search(retrieval_dict['query_' + from_modality], k)

        # Convert ids into labels
        topk_labels_list = []
        for topk_ids in topk_ids_list:
            topk_labels = list(map(lambda x: archive_labels[x], topk_ids))
            topk_labels_list.append(topk_labels)

        topk_labels_list = np.asarray(topk_labels_list)

        assert k == topk_labels_list.shape[1]
        num_queries = topk_labels_list.shape[0]

        # Calculate metrics
        precision = .0
        recall = .0
        f1 = .0
        for q, topk_labels in tqdm.tqdm(zip(query_labels, topk_labels_list)):

            # Accumulate per (q, topk-labels) pair
            total_q_prec = .0
            total_q_rec = .0
            total_q_f1 = .0

            for r in topk_labels:
                num_correct = np.logical_and(q, r).sum()
                prec = num_correct / r.sum()
                rec = num_correct / q.sum()

                total_q_prec += prec
                total_q_rec += rec
                total_q_f1 += (2*prec*rec) / (prec+rec) if num_correct else .0 

            precision += total_q_prec / k
            recall += total_q_rec / k
            f1 += total_q_f1 / k


        avg_precision = precision / num_queries
        avg_recall = recall / num_queries
        avg_f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)

        results[f"{from_modality}_{to_modality}"] = {
            'prec': avg_precision,
            'rec': avg_recall,
            'f1': avg_f1,
        }

    print("\nRetrieval performance")
    pprint.pprint(results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', type=str, help='<8-character-id> of model to be evaluated. Stored under /trained_models/<8-character-id>...')
    parser.add_argument('device', type=int, help='GPU number')
    
    args = parser.parse_args()

    main(args.model_id, args.device)
