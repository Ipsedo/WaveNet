from audio import WavDataset, to_wav
from networks import WaveNet
from generate import generate_from

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader

from statistics import mean

from tqdm import tqdm
from glob import glob

import argparse


def main() -> None:
    parser = argparse.ArgumentParser("Train WaveNet")

    parser.add_argument("--n-class", type=int, default=128)
    parser.add_argument("--n-block", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--hidden-channel", type=int, default=64)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--sample-length", type=int, default=32000)
    parser.add_argument("--generate", type=bool, default=True)

    parser.add_argument("audio_files", type=str)

    args = parser.parse_args()

    n_class = args.n_class
    n_block = args.n_block
    n_layer = args.n_layer
    hidden_channel = args.hidden_channel

    sample_rate = args.sample_rate
    sample_length = args.sample_length

    wavenet = WaveNet(
        n_block, n_layer, 1, hidden_channel, n_class
    )

    wavenet.cuda()

    optim = th.optim.Adam(wavenet.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    files = glob(args.audio_files)

    data_set = WavDataset(
        files, n_class,
        sample_rate, sample_length,
        True
    )

    data_loader = DataLoader(
        data_set, batch_size=4, shuffle=True
    )

    nb_epoch = 10

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

        if args.generate:
            with th.no_grad():
                n_sec_gen = 10
                n_sample = sample_rate * n_sec_gen
                init = th.rand(1024).cuda() * 2. - 1.

                generated_sound = generate_from(
                    wavenet, n_sample, init
                ).cpu()

                to_wav(
                    generated_sound.numpy(),
                    sample_rate,
                    f"./res/gen.wav"
                )


if __name__ == '__main__':
    main()
