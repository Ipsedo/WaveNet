from audio import WavDataset, to_wav
from networks import WaveNet
from generate import generate_from

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from glob import glob


def main() -> None:
    n_class = 128
    n_block = 3
    n_layer = 8
    hidden_channel = 64

    wavenet = WaveNet(
        n_block, n_layer, 1, hidden_channel, n_class
    )

    wavenet.cuda()

    optim = th.optim.Adam(wavenet.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    files = glob("./res/test_16000Hz/*.flac")

    data_set = WavDataset(files, n_class, 16000, 32000, True)

    data_loader = DataLoader(data_set, batch_size=4, shuffle=True)

    nb_epoch = 10

    for e in range(nb_epoch):

        tqdm_bar = tqdm(data_loader)
        for x, y in tqdm_bar:
            x = x.cuda()
            y = y.cuda()

            out = wavenet(x)

            loss = criterion(out, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            tqdm_bar.set_description(f"Epoch {e}, loss = {loss.item()}")

        with th.no_grad():
            n_sec_gen = 2
            n_sample = 16000 * n_sec_gen
            init = th.rand(100).cuda() * 2. - 1.

            generated_sound = generate_from(wavenet, n_sample, init).cpu()

            to_wav(
                generated_sound.numpy(),
                16000, f"./res/gen_epoch_{e}.wav"
            )


if __name__ == '__main__':
    main()
