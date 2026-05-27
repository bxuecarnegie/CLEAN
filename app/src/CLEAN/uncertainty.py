import torch
from .utils import * 
from .model import LayerNormNet
from .distance_map import *
from .evaluate import *
from .dataloader import *
import pandas as pd
import random
import warnings

def get_cluster_cen(model_emb_train, model_emb_test,
                      ec_id_dict_train, id_ec_test,
                      device, dtype, dot=False):
    '''
    Get the pair-wise distance map for test queries and train EC cluster centers
    map is of size of (N_test_ids, N_EC_train)
    '''
    print("The embedding sizes for train and test:",
          model_emb_train.size(), model_emb_test.size())
    # get cluster center for all EC appeared in training set
    cluster_center_model = get_cluster_center(
        model_emb_train, ec_id_dict_train)
    return cluster_center_model


def collect_valid_negatives(
    max_ec,
    ec_id_dict_train,
    id_ec_train,
    negative,
    neg_target,
    allow_global_fallback=True,
):
    """
    Collect valid negative proteins for one EC during GMM.

    Returns:
      neg_dict: dict[str, list[str]]
          protein_id -> EC labels
      neg_source: str
          one of {"hard", "global", "none"}
    """
    neg_target = int(neg_target)
    neg_dict = {}

    # Hard negative ECs
    hard_negative_ids = []

    if negative is not None and max_ec in negative:
        for neg_ec in negative[max_ec].get("negative", []):
            if neg_ec not in ec_id_dict_train:
                continue

            for protein_id in ec_id_dict_train[neg_ec]:
                if max_ec not in id_ec_train[protein_id]:
                    hard_negative_ids.append(protein_id)

    hard_negative_ids = list(set(hard_negative_ids))

    if len(hard_negative_ids) > 0:
        sample_size = min(neg_target, len(hard_negative_ids))
        sampled_ids = random.sample(hard_negative_ids, k=sample_size)

        for protein_id in sampled_ids:
            neg_dict[protein_id] = id_ec_train[protein_id]

        return neg_dict, "hard"

    # Global negative ECs fallback
    if allow_global_fallback:
        global_negative_ids = [
            protein_id
            for protein_id, ecs in id_ec_train.items()
            if max_ec not in ecs
        ]

        global_negative_ids = list(set(global_negative_ids))

        if len(global_negative_ids) > 0:
            sample_size = min(neg_target, len(global_negative_ids))
            sampled_ids = random.sample(global_negative_ids, k=sample_size)

            for protein_id in sampled_ids:
                neg_dict[protein_id] = id_ec_train[protein_id]

            return neg_dict, "global"

    return {}, "none"


def get_dist(max_ec, train_data, report_metrics = False, 
                 pretrained=True, model_name=None, target = 300, neg_target = 2000, negative = None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')

    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet(512, 128, device, dtype)
    
    if pretrained:
        try:
            checkpoint = torch.load('./data/pretrained/'+ train_data +'.pth')
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load('./data/model/'+ model_name +'.pth')
        except FileNotFoundError as error:
            raise Exception('No model found!')
            
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt')
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt')
    else:
        emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
    
    id_ec_test = {}
    for ids in ec_id_dict_train[max_ec]:
        #if len(id_ec_train[ids]) == 1:
        id_ec_test[ids] = max_ec
    
    # Precompute valid negatives
    neg_dict, neg_source = collect_valid_negatives(
        max_ec=max_ec,
        ec_id_dict_train=ec_id_dict_train,
        id_ec_train=id_ec_train,
        negative=negative,
        neg_target=neg_target,
        allow_global_fallback=True,
    )

    if neg_source == "global":
        print(
            f"[WARN] {max_ec}: no valid hard negatives found; "
            f"using {len(neg_dict)} global negatives instead."
        )
    elif neg_source == "none":
        print(
            f"[WARN] {max_ec}: no valid negatives found; "
            f"skipping negative-distance calculation for this EC."
        )
    else:
        print(f"[INFO] {max_ec}: using {len(neg_dict)} hard negatives.")

    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    ec_centers = get_cluster_cen(
        emb_train,
        emb_test,
        ec_id_dict_train,
        id_ec_test,
        device,
        dtype
    )

    distances = []
    for i in range(len(emb_test)):
        dist = (
            emb_test[i] - ec_centers[max_ec].to(device)
        ).norm(dim=0, p=2).detach().cpu().numpy().item()
        distances.append(dist)

    neg_distances = []

    if len(neg_dict) > 0:
        neg_emb_test = model_embedding_test(neg_dict, model, device, dtype)

        for i in range(len(neg_emb_test)):
            dist = (
                neg_emb_test[i] - ec_centers[max_ec].to(device)
            ).norm(dim=0, p=2).detach().cpu().numpy().item()
            neg_distances.append(dist)
    else:
        print(f"[WARN] {max_ec}: returning empty neg_distances.")

    return distances, neg_distances
