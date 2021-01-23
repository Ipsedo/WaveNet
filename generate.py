import torch as th
from networks import WaveNet
from audio import quantize, mu_encode, de_quantize


def generate_from(
        model: WaveNet, n_sample: int,
        initial_sample: th.Tensor) -> th.Tensor:
    assert len(initial_sample.size()) == 1, \
        f"initial_sample.size() must be length == 1, " \
        f"actual = {len(initial_sample.size())}"

    assert initial_sample.size(0) < n_sample, \
        f"initial_sample.size(0) must be < n_sample, " \
        f"actual {initial_sample.size(0)} < {n_sample}"

    generated_raw = th.zeros(n_sample)
    generated_raw[:initial_sample.size(0)] = initial_sample
    generated_raw = generated_raw.unsqueeze(0).unsqueeze(0)

    for sample_idx in range(initial_sample.size(0), n_sample):
        o = model(generated_raw).argmax(-1).squeeze(0)
        o_i = o[sample_idx]
        generated_raw[:, :, sample_idx] = de_quantize(o_i, model.n_class)

    return generated_raw


if __name__ == '__main__':
    wavenet = WaveNet(4, 9, 1, 32, 256)

    n_sample = 1024
    init = th.ones(1)

    gen = generate_from(wavenet, n_sample, init)

    print(init)
    print(gen)
