import torch as th
import torch.utils.data as th_data

import soundfile as sf

from tqdm import tqdm

from typing import List


def quantize(x: th.Tensor, n_class: int) -> th.Tensor:
    bins = th.linspace(-1, 1, n_class - 1).to(x.device)
    return th.bucketize(x, bins)


def de_quantize(x: th.Tensor, n_class: int) -> th.Tensor:
    bins = th.linspace(-1, 1, n_class).to(x.device)
    return th.gather(bins, 0, x)


def mu_encode(x: th.Tensor, mu: int) -> th.Tensor:
    return th.sign(x) * th.log(1. + mu * th.abs(x)) / \
           th.log(th.tensor(1. + mu).to(x.device))


def mu_decode(x: th.Tensor, mu: int) -> th.Tensor:
    return th.sign(x) * (1. / mu) * ((1. + mu) ** th.abs(x) - 1.)


def to_wav(raw_audio: th.Tensor, sample_rate: int, out_file: str) -> None:
    sf.write(out_file, raw_audio, sample_rate)


def _get_sample_number(
        wav_paths: List[str], sample_rate: int,
        sample_length: int, progress_bar: bool
) -> int:
    sample_nb = 0

    progress = tqdm(wav_paths) if progress_bar else wav_paths

    for w_p in progress:
        raw_audio, curr_sr = sf.read(w_p)

        if curr_sr != sample_rate:
            raise Exception(
                f"Wrong sample rate : "
                f"needed = {sample_rate}, actual = {curr_sr}"
            )

        sample_nb += raw_audio.shape[0] // sample_length

    return sample_nb


def read_wav(wav_path: str, sample_rate: int) -> th.Tensor:
    raw_audio, curr_sr = sf.read(wav_path)

    if curr_sr != sample_rate:
        raise Exception(
            f"Wrong sample rate : "
            f"needed = {sample_rate}, actual = {curr_sr}"
        )

    return th.from_numpy(raw_audio)


class WavDataset(th_data.TensorDataset):
    def __init__(
            self, wav_paths: List[str],
            n_class: int, sample_rate: int,
            sample_length: int,
            progress_bar: bool = True
    ):
        sample_nb = _get_sample_number(
            wav_paths, sample_rate, sample_length + 1, progress_bar
        )

        x = th.empty(sample_nb, 1, sample_length)
        y = th.empty(sample_nb, sample_length, dtype=th.long)

        sample_idx = 0

        progress = tqdm(wav_paths) if progress_bar else wav_paths

        for w_p in progress:
            raw_audio = read_wav(w_p, sample_rate)

            to_keep = raw_audio.size(0) - \
                      raw_audio.size(0) % (sample_length + 1)

            raw_audio = raw_audio[:to_keep, :].mean(dim=-1) \
                if len(raw_audio.size()) > 1 \
                else raw_audio[:to_keep]

            samples = th.stack(raw_audio.split(sample_length + 1, dim=0), dim=0)

            samples = mu_encode(samples, n_class)

            curr_x = samples[:, :-1]
            curr_y = quantize(samples, n_class)[:, 1:]

            x[sample_idx:sample_idx + curr_x.size(0), 0, :] = curr_x
            y[sample_idx:sample_idx + curr_y.size(0), :] = curr_y

            sample_idx += curr_x.size(0)

        super(WavDataset, self).__init__(x, y)


if __name__ == '__main__':
    x = th.rand(10) * 2 - 1

    print(x)

    m_x = mu_encode(x, 256)
    print(m_x)

    q_x = quantize(x, 256)
    print(q_x)

    d_q_x = de_quantize(q_x, 256)
    print(d_q_x)

    musics = [
        "/home/samuel/Documents/WaveNet/res/test_16000Hz/01 satie - musiques intimes et secretes - nostalgie.flac"
    ]

    dataset = WavDataset(musics, 256, 16000, 32000, progress_bar=True)

    x, y = dataset[2]

    print(x.size())
    print(y.size())

    print(x[0, :20])
    print(de_quantize(y[:20], 256))
