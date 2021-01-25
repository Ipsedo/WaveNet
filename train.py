from audio import WavDataset, to_wav
from networks import WaveNet
from generate import generate_from

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader

import mlflow

from statistics import mean

from tqdm import tqdm
from glob import glob

import os
from os.path import join

import argparse


def main() -> None:
    # argparse stuff
    parser = argparse.ArgumentParser("Train WaveNet")

    parser.add_argument("--nb-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)

    parser.add_argument("--n-class", type=int, default=128)
    parser.add_argument("--n-block", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--hidden-channel", type=int, default=160)
    parser.add_argument("--residual-channel", type=int, default=48)

    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--sample-length", type=int, default=32000)

    parser.add_argument("--gen-sec", type=float, default=0.)

    parser.add_argument("mlflow_run_id", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("audio_files", type=str)

    args = parser.parse_args()

    output_dir = args.output_dir

    # create output dir
    if not os.path.exists(output_dir):
        if os.path.isfile(output_dir):
            raise NotADirectoryError(f"{output_dir} is not a directory !")
        else:
            os.mkdir(output_dir)

    # MLFlow stuff
    mlflow.set_experiment("WaveNet")

    mlflow.start_run(run_name=f"train_{args.mlflow_run_id}")

    # hyperparam init
    n_class = args.n_class
    n_block = args.n_block
    n_layer = args.n_layer
    residual_channel = args.residual_channel
    hidden_channel = args.hidden_channel

    nb_epoch = args.nb_epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    sample_rate = args.sample_rate
    sample_length = args.sample_length

    # init wavenet model
    wavenet = WaveNet(
        n_block, n_layer,
        1, residual_channel, hidden_channel,
        n_class
    )

    wavenet.cuda()

    optim = th.optim.Adam(
        wavenet.parameters(),
        lr=learning_rate
    )

    criterion = nn.CrossEntropyLoss()

    files = glob(args.audio_files)

    data_set = WavDataset(
        files, n_class,
        sample_rate, sample_length,
        True
    )

    data_loader = DataLoader(
        data_set, batch_size=batch_size, shuffle=True
    )

    # set mlflow params
    mlflow.log_params({
        "n_class": n_class,
        "n_block": n_block,
        "n_layer": n_layer,
        "residual_channel": residual_channel,
        "hidden_channel": hidden_channel,
        "sample_rate": sample_rate,
        "sample_length": sample_length,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "nb_epoch": nb_epoch,
        "output_dir": output_dir
    })

    idx = 0

    # train loop
    for e in range(nb_epoch):

        tqdm_bar = tqdm(data_loader)

        losses = []

        for x, y in tqdm_bar:
            x = x.cuda()
            y = y.cuda()

            out = wavenet(x)

            loss = criterion(out, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())

            tqdm_bar.set_description(
                f"Epoch {e}, "
                f"loss = {mean(losses)}"
            )

            if idx % 250 == 0:
                mlflow.log_metric(
                    "loss", loss.item(),
                    step=idx
                )

            idx += 1

        if args.gen_sec > 0.:
            with th.no_grad():
                n_sample = int(sample_rate * args.gen_sec)
                init = th.rand(4096).cuda() * 2. - 1.

                generated_sound = generate_from(
                    wavenet, n_sample, init
                ).cpu()

                to_wav(
                    generated_sound.numpy(),
                    sample_rate,
                    join(output_dir, f"gen_epoch_{e}.wav")
                )

                mlflow.log_artifact(
                    join(output_dir, f"gen_epoch_{e}.wav"),
                    f"gen_epoch_{e}.wav"
                )

        th.save(
            wavenet.state_dict(),
            join(output_dir, f"wavenet_epoch_{e}.pt")
        )
        th.save(
            optim.state_dict(),
            join(output_dir, f"optim_epoch_{e}.pt")
        )

        mlflow.log_artifact(
            join(output_dir, f"wavenet_epoch_{e}.pt")
        )
        mlflow.log_artifact(
            join(output_dir, f"optim_epoch_{e}.pt")
        )


if __name__ == '__main__':
    main()
