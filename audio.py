import torch as th


def quantize(x: th.Tensor, n_class: int) -> th.Tensor:
    bins = th.linspace(-1, 1, n_class - 1)
    return th.bucketize(x, bins)


def de_quantize(x: th.Tensor, n_class: int) -> th.Tensor:
    bins = th.linspace(-1, 1, n_class)
    return th.gather(bins, 0, x)


def mu_encode(x: th.Tensor, mu: int) -> th.Tensor:
    return th.sign(x) * th.log(1. + mu * th.abs(x)) / th.log(th.tensor(1. + mu))


if __name__ == '__main__':
    x = th.rand(10) * 2 - 1

    print(x)

    m_x = mu_encode(x, 256)
    print(m_x)

    q_x = quantize(x, 256)
    print(q_x)

    d_q_x = de_quantize(q_x, 256)
    print(d_q_x)
