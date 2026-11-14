"""Train either core model on user-provided normal measurements."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("PGAE-tau", "ConvAE"), default="PGAE-tau")
    parser.add_argument("--case", choices=("case14", "case30", "case57"), default="case30")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")
    os.environ["PGAE_CASE"] = args.case
    if args.device is not None:
        os.environ["PGAE_DEVICE"] = args.device

    import random
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from configs.nn_setting import nn_setting
    from data.dataset import rnn_dataset
    from models.detection.pgae_tau import PGAE_tau
    from models.baselines.convae import Conv_AE

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    nn_setting.update(epochs=args.epochs, batch_size=args.batch_size)
    # Match the case14 backbone expected by the localization implementation.
    if args.model == "ConvAE" and args.case == "case14":
        nn_setting["lattent_dim"] = 10
    train_set = rnn_dataset("train", True)
    valid_set = rnn_dataset("valid", True)
    model = PGAE_tau() if args.model == "PGAE-tau" else Conv_AE()
    model.to(nn_setting["device"])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    model.fit(train_loader, valid_loader)
    print(f"Saved best checkpoint: {model.path}")


if __name__ == "__main__":
    main()
