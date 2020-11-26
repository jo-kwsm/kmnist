import argparse
from glob import glob
import os

import pandas as pd


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
                    compare all results.
                    """
    )
    parser.add_argument("result", type=str, default=None, help="path of result directry.")

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    final_result = pd.DataFrame(
        columns=[
            "experiment_name",
            "val_loss",
            "val_acc@1",
            "val_f1score",
            "test_loss",
            "test_acc@1",
            "test_f1score",
        ]
    )

    for dir_path in glob(os.path.join(args.result, "*")):
        experiment_name = os.path.basename(dir_path)
        try:    
            val_log = pd.read_csv(os.path.join(dir_path, "validation_log.csv"))
            test_log = pd.read_csv(os.path.join(dir_path, "test_log.csv"))
            tmp = pd.Series(
                [
                    experiment_name,
                    val_log["loss"][0],
                    val_log["acc@1"][0],
                    val_log["f1score"][0],
                    test_log["loss"][0],
                    test_log["acc@1"][0],
                    test_log["f1score"][0],
                ],
                index=final_result.columns,
            )
            final_result = final_result.append(tmp, ignore_index=True)
        except:
            pass
        
    final_result.to_csv(os.path.join(args.result, "final_result.csv"), index=False)


if __name__ == "__main__":
    main()
