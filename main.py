import argparse
import os
import random as rd
import shutil

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from data.dataset import PoisonLabelDataset
from data.utils import (
    gen_poison_idx,
    get_bd_transform,
    get_dataset,
    get_loader,
    get_transform,
)
from model.model import LinearModel
from model.utils import (
    get_network,
    get_optimizer,
    get_scheduler,
    load_state,
)
from utils.setup import (
    get_logger,
    get_saved_dir,
    get_storage_dir,
    load_config,
    set_seed,
)
from utils.trainer.semi import linear_test


def main():
    print("===Setup running===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/compute_efficient_v4_1026.yaml")
    parser.add_argument("--gpu", default="4", type=str)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="checkpoint name (empty string means the latest checkpoint)\
            or False (means training from scratch).",
    )
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-port",
        default="23456",
        type=str,
        help="port used to set up distributed training",
    )
    args = parser.parse_args()

    config, inner_dir, config_name = load_config(args.config)
    args.saved_dir, args.log_dir = get_saved_dir(
        config, inner_dir, config_name, args.resume
    )
    shutil.copy2(args.config, args.saved_dir)
    args.storage_dir, args.ckpt_dir, _ = get_storage_dir(
        config, inner_dir, config_name, args.resume
    )
    shutil.copy2(args.config, args.storage_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        args.distributed = True
    else:
        args.distributed = False
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        print("Distributed training on GPUs: {}.".format(args.gpu))
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, config),
        )
    else:
        print("Training on a single GPU: {}.".format(args.gpu))
        main_worker(0, ngpus_per_node, args, config)


def main_worker(gpu, ngpus_per_node, args, config):
    set_seed(**config["seed"])
    logger = get_logger(args.log_dir, "asd.log", args.resume, gpu == 0)
    torch.cuda.set_device(gpu)
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:{}".format(args.dist_port),
            world_size=args.world_size,
            rank=args.rank,
        )
        logger.warning("Only log rank 0 in distributed training!")

    logger.info("===Prepare data===")
    bd_config = config["backdoor"]
    logger.info("Load backdoor config:\n{}".format(bd_config))
    bd_transform = get_bd_transform(bd_config)
    target_label = bd_config["target_label"]
    poison_ratio = bd_config["poison_ratio"]

    pre_transform = get_transform(config["transform"]["pre"])
    train_primary_transform = get_transform(config["transform"]["train"]["primary"])
    train_remaining_transform = get_transform(config["transform"]["train"]["remaining"])
    train_transform = {
        "pre": pre_transform,
        "primary": train_primary_transform,
        "remaining": train_remaining_transform,
    }
    logger.info("Training transformations:\n {}".format(train_transform))
    test_primary_transform = get_transform(config["transform"]["test"]["primary"])
    test_remaining_transform = get_transform(config["transform"]["test"]["remaining"])
    test_transform = {
        "pre": pre_transform,
        "primary": test_primary_transform,
        "remaining": test_remaining_transform,
    }
    logger.info("Test transformations:\n {}".format(test_transform))

    logger.info("Load dataset from: {}".format(config["dataset_dir"]))
    clean_train_data = get_dataset(
        config["dataset_dir"], train_transform, prefetch=config["prefetch"]
    )
    clean_test_data = get_dataset(
        config["dataset_dir"], test_transform, train=False, prefetch=config["prefetch"]
    )

    poison_idx_path = os.path.join(args.saved_dir, "poison_idx.npy")
    if os.path.exists(poison_idx_path):
        poison_train_idx = np.load(poison_idx_path)
        logger.info("Load poisoned index to {}".format(poison_idx_path))
    else:
        poison_train_idx = gen_poison_idx(clean_train_data, target_label, poison_ratio)
        np.save(poison_idx_path, poison_train_idx)
        logger.info("Save poisoned index to {}".format(poison_idx_path))

    poison_train_data = PoisonLabelDataset(
        clean_train_data, bd_transform, poison_train_idx, target_label
    )
    poison_test_idx = gen_poison_idx(clean_test_data, target_label)
    poison_test_data = PoisonLabelDataset(
        clean_test_data, bd_transform, poison_test_idx, target_label
    )

    poison_train_loader = get_loader(poison_train_data, config["loader"], shuffle=True)
    clean_test_loader = get_loader(clean_test_data, config["loader"])
    poison_test_loader = get_loader(poison_test_data, config["loader"])

    logger.info("\n===Setup training===")
    backbone = get_network(config["network"])
    logger.info("Create network: {}".format(config["network"]))
    linear_model = LinearModel(backbone, backbone.feature_dim, config["num_classes"])
    linear_model = linear_model.cuda(gpu)
    if args.distributed:
        linear_model = DistributedDataParallel(linear_model, device_ids=[gpu])

    optimizer = get_optimizer(linear_model, config["optimizer"])
    logger.info("Create optimizer: {}".format(optimizer))

    scheduler = get_scheduler(optimizer, config["lr_scheduler"])
    logger.info("Create scheduler: {}".format(config["lr_scheduler"]))
    load_state(
        linear_model,
        args.resume,
        args.ckpt_dir,
        gpu,
        logger,
        optimizer,
        scheduler,
        is_best=True,
    )

    criterion = torch.nn.CrossEntropyLoss().cuda()
    for _ in range(40):
        linear_model.train()
        for batch_idx, batch in enumerate(poison_train_loader):
            data = batch["img"].cuda(gpu, non_blocking=True)
            target = batch["target"].cuda(gpu, non_blocking=True)
            optimizer.zero_grad()
            output = linear_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    logger.info("Test model on clean data...")
    linear_test(linear_model, clean_test_loader, criterion, logger)
    logger.info("Test model on poison data...")
    linear_test(linear_model, poison_test_loader, criterion, logger)

    # RD
    times = 30
    ratio = 0.5
    decay = 0.93
    troj_list = []
    out_list = []
    for i in range(len(poison_train_idx)):
        if poison_train_idx[i] == 1:
            troj_list.append(i)
        else:
            out_list.append(i)
    for _ in range(times):
        poison_train_idx = np.zeros(len(clean_train_data))
        for i in troj_list:
            poison_train_idx[i] = 1
        linear_model = LinearModel(backbone, backbone.feature_dim, config["num_classes"])
        linear_model = linear_model.cuda(gpu)
        if args.distributed:
            linear_model = DistributedDataParallel(linear_model, device_ids=[gpu])
        optimizer = get_optimizer(linear_model, config["optimizer"])
        loss_func = torch.nn.CrossEntropyLoss()
        poison_train_data = PoisonLabelDataset(
            clean_train_data, bd_transform, poison_train_idx, target_label
        )
        poison_train_loader = get_loader(poison_train_data, config["loader"], shuffle=True)
        for _ in range(6):
            linear_model.train()
            for batch_idx, batch in enumerate(poison_train_loader):
                data = batch["img"].cuda(gpu, non_blocking=True)
                target = batch["target"].cuda(gpu, non_blocking=True)
                optimizer.zero_grad()
                output = linear_model(data)
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()
        poison_eval_loader = get_loader(poison_train_data, config["loader"], shuffle=False)
        linear_model.eval()
        scores = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(poison_eval_loader):
                data = batch["img"].cuda(gpu, non_blocking=True)
                target = batch["target"].cuda(gpu, non_blocking=True)
                output = linear_model(data)
                prob = torch.nn.Softmax()(output)
                for i in range(prob.shape[0]):
                    if batch["poison"][i] == 1:
                        one_hot = torch.zeros_like(output[i])
                        one_hot[int(target[i])] = 1
                        score = torch.linalg.norm(prob[i] - one_hot).item()
                        scores.append(score)
        print(len(scores))
        score_list = np.argsort(np.array(scores))
        lens = len(troj_list)
        diet_list = [troj_list[i] for i in score_list[0:int(len(scores) * ratio + 0.5)]]
        troj_list = [i for i in troj_list if i not in diet_list]
        add_list = rd.sample(out_list, int(lens * ratio + 0.5))
        out_list = [i for i in out_list if i not in add_list]
        out_list.extend(diet_list)
        troj_list.extend(add_list)
        ratio *= decay

    # eval RD
    backbone = get_network(config["network"])
    logger.info("Create network: {}".format(config["network"]))
    linear_model = LinearModel(backbone, backbone.feature_dim, config["num_classes"])
    linear_model = linear_model.cuda(gpu)
    if args.distributed:
        linear_model = DistributedDataParallel(linear_model, device_ids=[gpu])

    optimizer = get_optimizer(linear_model, config["optimizer"])
    criterion = torch.nn.CrossEntropyLoss().cuda()
    poison_train_idx = np.zeros(len(clean_train_data))
    for i in troj_list:
        poison_train_idx[i] = 1
    poison_train_data = PoisonLabelDataset(
        clean_train_data, bd_transform, poison_train_idx, target_label
    )
    poison_train_loader = get_loader(poison_train_data, config["loader"], shuffle=True)
    for _ in range(40):
        linear_model.train()
        for batch_idx, batch in enumerate(poison_train_loader):
            data = batch["img"].cuda(gpu, non_blocking=True)
            target = batch["target"].cuda(gpu, non_blocking=True)
            optimizer.zero_grad()
            output = linear_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    logger.info("Test model on clean data...")
    linear_test(linear_model, clean_test_loader, criterion, logger)
    logger.info("Test model on poison data...")
    linear_test(linear_model, poison_test_loader, criterion, logger)

if __name__ == "__main__":
    main()
