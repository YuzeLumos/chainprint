"""Support-fitted staged integer filtering; no renderer or operation metadata."""
import torch
from torch import nn
from torch.nn import functional as F


class QuantizedSeparableResponse(nn.Module):
    def __init__(self, device):
        super().__init__()
        radius = torch.arange(5, device=device, dtype=torch.float32)
        self.kernel_logits = nn.Parameter(-radius.square() / (2 * .5 ** 2))
        self.gain = nn.Parameter(torch.ones((), device=device))
        self.threshold = nn.Parameter(torch.zeros((), device=device))

    @staticmethod
    def quantize(value, surrogate):
        return value + (value.round() - value).detach() if surrogate else value.round()

    def lowpass(self, value, surrogate=False):
        logits = torch.cat((self.kernel_logits[1:].flip(0), self.kernel_logits))
        weights = logits.softmax(0)
        current = value[None]
        for vertical in (False, True):
            shape = (1, 1, 9, 1) if vertical else (1, 1, 1, 9)
            padding = (0, 0, 4, 4) if vertical else (4, 4, 0, 0)
            kernel = weights.reshape(shape).expand(3, -1, -1, -1).contiguous()
            for _ in range(3):
                current = self.quantize(F.conv2d(F.pad(current, padding, mode='replicate'), kernel, groups=3), surrogate)
        return current[0]

    def forward(self, value, surrogate=False):
        delta = self.lowpass(value, surrogate) - value
        threshold = self.threshold.clamp_min(0)
        gate = (delta.abs() > threshold).to(delta.dtype)
        if surrogate:
            smooth = torch.sigmoid(delta.abs() - threshold)
            gate = smooth + (gate - smooth).detach()
        result = (value + self.gain * delta * gate).clamp(0, 255)
        return self.quantize(result, surrogate)

    @classmethod
    def fit_support(cls, before, after, mask, device):
        model = cls(device)
        x = torch.as_tensor(before, device=device, dtype=torch.float32).mul(255).round()
        y = torch.as_tensor(after, device=device, dtype=torch.float32).mul(255).round()
        with torch.no_grad():
            delta = model.lowpass(x) - x
            denominator = delta[mask].square().sum().clamp_min(1e-12)
            model.gain.copy_((delta[mask] * (y - x)[mask]).sum() / denominator)
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        last_loss = None
        # Single fixed terminal iterate, never selected on held pixels.
        with torch.enable_grad():
            for _ in range(128):
                optimizer.zero_grad(set_to_none=True)
                predicted = model(x, surrogate=True)
                loss = (predicted[mask] - y[mask]).square().mean()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    model.threshold.clamp_(min=0)
                last_loss = loss.detach()
        model.eval()
        return model, float(last_loss)

    @torch.no_grad()
    def predict(self, before):
        byte = (before.to(torch.float32) * 255).round()
        return self(byte).to(before.dtype) / 255
