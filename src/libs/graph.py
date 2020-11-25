import argparse
import dataclasses

import pandas as pd
import matplotlib.pyplot as plt


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
                    make graphs for trainning logs.
                    """
    )
    parser.add_argument("log", type=str, default=None, help="path of a log file")

    return parser.parse_args()


def make_line(data_name:str, data:pd.DataFrame, save_dir:str) -> None:
    plt.figure()
    plt.plot(data["train_" + data_name], label="train")
    plt.plot(data["val_" + data_name], label="val")
    plt.xlabel("epoch")
    plt.ylabel(data_name)
    plt.legend()
    save_path = save_dir + data_name + ".png"
    plt.savefig(save_path)


def make_graphs(log_path:str) -> None:
    logs = pd.read_csv(log_path)
    save_dir = "/".join(log_path.split("/")[:-1]) + "/"
    make_line("loss", logs, save_dir)
    make_line("acc@1", logs, save_dir)


def main() -> None:
    args = get_arguments()
    make_graphs(args.log)


if __name__ == "__main__":
    main()
