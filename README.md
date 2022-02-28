Implementation of [Model-Based Reinforcement Learning for Atari](https://arxiv.org/abs/1903.00374).

Installing gym with atari.

```bash
pip install gym[atari,accept-rom-license]
```

Fixed atari wrappers so that the shape of a vectorized atari environment state is (n_envs, 105, 80, 3)

