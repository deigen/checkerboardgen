import torch
import torch.nn as nn

class AdaLNModulation(torch.nn.Module):
    def __init__(self, channels_x, channels_y=None, with_gate=True, gate_init=0.0):
        super().__init__()
        channels_y = channels_y if channels_y is not None else channels_x
        self.with_gate = with_gate
        assert isinstance(with_gate, bool)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels_y, (2 + int(with_gate)) * channels_x, bias=True)
        )
        # init modulation to 0
        torch.nn.init.zeros_(self.adaLN_modulation[-1].weight)
        torch.nn.init.constant_(self.adaLN_modulation[-1].bias[:channels_x], 0.0)  # shift
        torch.nn.init.constant_(self.adaLN_modulation[-1].bias[channels_x:2*channels_x], 1.0)  # scale
        if with_gate:
            torch.nn.init.constant_(self.adaLN_modulation[-1].bias[2*channels_x:], gate_init)
        self.adaLN_modulation[-1]._is_hf_initialized = True  # flag as init done

    def forward(self, x, y):
        if self.with_gate:
            shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
            x = x * scale + shift
            return x, gate
        else:
            shift, scale = self.adaLN_modulation(y).chunk(2, dim=-1)
            x = x * scale + shift
            return x
