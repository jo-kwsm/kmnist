import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import wandb
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomAffine,
    Resize,
    ToTensor,
)

from libs.checkpoint import resume, save_checkpoint
from libs.class_id_map import get_cls2id_map
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.helper import evaluate, train
from libs.graph import make_graphs
from libs.loss_fn import get_criterion
from libs.mean_std import get_mean, get_std
from libs.models import get_model

random_seed = 1234


"""
Copyright (c) 2020 yiskw713
"""


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        train a network for image classification with KMNIST Dataset.
        """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Add --no_wandb option if you do not want to use wandb.",
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # configuration
    config = get_config(args.config)

    # save log files in the directory which contains config file.
    result_path = os.path.dirname(args.config)
    experiment_name = os.path.basename(result_path)

    # if a experiment has already done, train won't start.
    if os.path.exists(os.path.join(result_path, "final_model.prm")):
        print("Already done.")
        return

    # cpu or cuda
    device = get_device(allow_only_gpu=True)

    # Dataloader
    train_transform = Compose(
        [
            Resize(config.size),
            RandomAffine(degrees=config.degrees, translate=(config.translate, config.translate)),
            ToTensor(),
            Normalize(mean=get_mean(), std=get_std()),
        ]
    )

    val_transform = Compose(
        [
            Resize(config.size),
            ToTensor(),
            Normalize(mean=get_mean(), std=get_std())
        ]
    )

    imgs = np.load(config.train_imgs)["arr_0"]
    imgs = imgs.reshape(-1,28,28)
    ids = np.load(config.train_ids)["arr_0"]
    train_imgs, val_imgs, train_ids, val_ids = train_test_split(imgs, ids, test_size=0.1, random_state=random_seed, stratify=ids)

    train_loader = get_dataloader(
        imgs=train_imgs,
        ids=train_ids,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        transform=train_transform,
    )

    val_loader = get_dataloader(
        imgs=val_imgs,
        ids=val_ids,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        transform=val_transform,
    )

    # the number of classes
    n_classes = len(get_cls2id_map())

    # define a model
    model = get_model(config.model, n_classes, pretrained=config.pretrained)

    # send the model to cuda/cpu
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # keep training and validation log
    begin_epoch = 0
    best_loss = float("inf")
    log = pd.DataFrame(
        columns=[
            "epoch",
            "lr",
            "train_time[sec]",
            "train_loss",
            "train_acc@1",
            "train_f1s",
            "val_time[sec]",
            "val_loss",
            "val_acc@1",
            "val_f1s",
        ]
    )

    # resume if you want
    if args.resume:
        resume_path = os.path.join(result_path, "checkpoint.pth")
        begin_epoch, model, optimizer, best_loss = resume(resume_path, model, optimizer)

        log_path = os.path.join(result_path, "log.csv")
        assert os.path.exists(log_path), "there is no checkpoint at the result folder"
        log = pd.read_csv(log_path)

    # criterion for loss
    criterion = get_criterion(config.use_class_weight, train_ids, device)

    # Weights and biases
    if not args.no_wandb:
        wandb.init(
            name=experiment_name,
            config=config,
            project="image_classification_template",
            job_type="training",
            dirs="./wandb_result/",
        )
        # Magic
        wandb.watch(model, log="all")

    # train and validate model
    print("---------- Start training ----------")

    for epoch in range(begin_epoch, config.max_epoch):
        # training
        start = time.time()
        train_loss, train_acc1, train_f1s = train(
            train_loader, model, criterion, optimizer, epoch, device
        )
        train_time = int(time.time() - start)

        # validation
        start = time.time()
        val_loss, val_acc1, val_f1s, c_matrix = evaluate(
            val_loader, model, criterion, device
        )
        val_time = int(time.time() - start)

        # save a model if top1 acc is higher than ever
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(result_path, "best_model.prm"),
            )

        # save checkpoint every epoch
        save_checkpoint(result_path, epoch, model, optimizer, best_loss)

        # write logs to dataframe and csv file
        tmp = pd.Series(
            [
                epoch,
                optimizer.param_groups[0]["lr"],
                train_time,
                train_loss,
                train_acc1,
                train_f1s,
                val_time,
                val_loss,
                val_acc1,
                val_f1s,
            ],
            index=log.columns,
        )

        log = log.append(tmp, ignore_index=True)
        log.to_csv(os.path.join(result_path, "log.csv"), index=False)
        make_graphs(os.path.join(result_path, "log.csv"))

        # save logs to wandb
        if not args.no_wandb:
            wandb.log(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_time[sec]": train_time,
                    "train_loss": train_loss,
                    "train_acc@1": train_acc1,
                    "train_f1s": train_f1s,
                    "val_time[sec]": val_time,
                    "val_loss": val_loss,
                    "val_acc@1": val_acc1,
                    "val_f1s": val_f1s,
                },
                step=epoch,
            )

        print(
            """epoch: {}\tepoch time[sec]: {}\tlr: {}\ttrain loss: {:.4f}\t\
            val loss: {:.4f} val_acc1: {:.5f}\tval_f1s: {:.5f}
            """.format(
                epoch,
                train_time + val_time,
                optimizer.param_groups[0]["lr"],
                train_loss,
                val_loss,
                val_acc1,
                val_f1s,
            )
        )

    # save models
    torch.save(model.state_dict(), os.path.join(result_path, "final_model.prm"))

    # delete checkpoint
    os.remove(os.path.join(result_path, "checkpoint.pth"))

    print("Done")


if __name__ == "__main__":
    main()
