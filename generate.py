import torch as th
from networks import WaveNet
from audio import de_quantize, to_wav, mu_encode, mu_decode

from tqdm import tqdm


def generate_from(
        model: WaveNet, n_sample: int,
        initial_sample: th.Tensor) -> th.Tensor:
    assert len(initial_sample.size()) == 1, \
        f"initial_sample.size() must be length == 1, " \
        f"actual = {len(initial_sample.size())}"

    assert initial_sample.size(0) < n_sample, \
        f"initial_sample.size(0) must be < n_sample, " \
        f"actual {initial_sample.size(0)} < {n_sample}"

    generated_raw = th.zeros(initial_sample.size(0)).to(initial_sample.device)
    generated_raw[:initial_sample.size(0)] = mu_encode(initial_sample, model.n_class)
    generated_raw = generated_raw.unsqueeze(0).unsqueeze(0)

    for sample_idx in tqdm(range(n_sample - initial_sample.size(0))):
        o = model(generated_raw)

        o = o.permute(0, 2, 1).argmax(-1).squeeze(0)
        o = o[sample_idx]

        dq_o = de_quantize(o, model.n_class)

        generated_raw = th.cat(
            [generated_raw, dq_o[None, None, None]], dim=-1
        )

    return mu_decode(generated_raw.view(-1).detach(), model.n_class)


if __name__ == '__main__':
    wavenet = WaveNet(4, 9, 1, 32, 256).cuda()

    n_sample = 10
    init = th.ones(2).cuda()

    gen = generate_from(wavenet, n_sample, init).cpu()

    print(init)
    print(gen)

    to_wav(gen, 16000, "./res/test.wav")
