from ncps.mini_keras import optimizers
from ncps.mini_keras.backend.torch.optimizers import torch_adam


class AdamW(torch_adam.Adam, optimizers.AdamW):
    pass
