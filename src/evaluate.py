import argparse
import csv
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from libs.class_id_map import get_cls2id_map
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.helper import evaluate
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
                    train a network for image classification
                    with KMNIST Dataset
                    """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument("mode", type=str, help="validation or test")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="""path to the trained model. If you do not specify, the trained model,
            'best_acc1_model.prm' in result directory will be used.""",
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # configuration
    config = get_config(args.config)

    result_path = os.path.dirname(args.config)

    # cpu or cuda
    device = get_device(allow_only_gpu=True)

    # Dataloader
    assert args.mode in ["validation", "test"]

    transform = Compose(
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
    test_imgs = np.load(config.test_imgs)["arr_0"]
    test_imgs = test_imgs.reshape(-1,28,28)
    test_ids = np.load(config.test_ids)["arr_0"]

    loader = get_dataloader(
        imgs=val_imgs if args.mode == "validation" else test_imgs,
        ids=val_ids if args.mode == "validation" else test_ids,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        transform=transform,
    )

    # the number of classes
    n_classes = len(get_cls2id_map())

    model = get_model(config.model, n_classes, pretrained=config.pretrained)

    # send the model to cuda/cpu
    model.to(device)

    # load the state dict of the model
    if args.model is not None:
        state_dict = torch.load(args.model)
    else:
        state_dict = torch.load(os.path.join(result_path, "best_model.prm"))

    model.load_state_dict(state_dict)

    # criterion for loss
    criterion = get_criterion(config.use_class_weight, train_ids, device)

    # train and validate model
    print(f"---------- Start evaluation for {args.mode} data ----------")

    # evaluation
    loss, acc1, f1s, c_matrix = evaluate(loader, model, criterion, device)

    print("loss: {:.5f}\tacc1: {:.2f}\tF1 Score: {:.2f}".format(loss, acc1, f1s))

    df = pd.DataFrame(
        {"loss": [loss], "acc@1": [acc1], "f1score": [f1s]},
        columns=["loss", "acc@1", "f1score"],
        index=None,
    )

    df.to_csv(os.path.join(result_path, "{}_log.csv").format(args.mode), index=False)

    with open(
        os.path.join(result_path, "{}_c_matrix.csv").format(args.mode), "w"
    ) as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(c_matrix)

    print("Done.")


if __name__ == "__main__":
    main()
