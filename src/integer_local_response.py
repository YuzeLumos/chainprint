"""An anonymous, single-support integer local-response family; no renderer API."""
from __future__ import annotations

import hashlib
import json
import time

import torch
from torch import nn
from torch.nn import functional as F

Q = 1 << 24
RHO_MAX = 3.999
START_RADII = (0.125, 0.5)
STEPS = 128
INITIAL_LR = 0.03


class IntegerLocalResponse(nn.Module):
    def __init__(self, device, rho=START_RADII[0]):
        super().__init__()
        device = torch.device(device)
        if device.type != "cpu":
            raise ValueError("IntegerLocalResponse uses the fixed CPU numerical environment")
        self.rho = nn.Parameter(torch.tensor(float(rho), device=device, dtype=torch.float64))
        self.gain = nn.Parameter(torch.ones((), device=device, dtype=torch.float64))
        self.tau = nn.Parameter(torch.zeros((), device=device, dtype=torch.float64))

    @staticmethod
    def _quantize(value, rule, surrogate):
        hard = rule(value)
        # Parenthesized zero has exactly zero forward value; do not reconstruct
        # hard values by subtracting and adding large fixed-point quantities.
        return hard.detach() + (value - value.detach()) if surrogate else hard

    def _kernel(self, surrogate=False):
        rho = self.rho.clamp(0, RHO_MAX)
        n = int(torch.floor(rho.detach()).item())
        weight = self._quantize(Q / (2 * rho + 1), torch.floor, surrogate)
        fringe = self._quantize((Q - (2 * n + 1) * weight) / 2, torch.floor, surrogate)
        return n, weight, fringe

    def _horizontal_pass(self, value, n, weight, fringe, surrogate):
        width = value.shape[-1]
        padded = F.pad(value[None], (n + 1, n + 1, 0, 0), mode="replicate")[0]
        prefix = torch.cat((torch.zeros_like(padded[..., :1]), padded.cumsum(dim=-1)), dim=-1)
        bulk = prefix[..., 2*n+2:2*n+2+width] - prefix[..., 1:1+width]
        edges = padded[..., :width] + padded[..., 2*n+2:2*n+2+width]
        total = bulk * weight + edges * fringe
        return self._quantize((total + Q // 2) / Q, torch.floor, surrogate)

    def lowpass(self, value, surrogate=False):
        current = value.to(dtype=torch.float64)
        n, weight, fringe = self._kernel(surrogate)
        for vertical in (False, True):
            if vertical:
                current = current.transpose(-1, -2)
            for _ in range(3):
                current = self._horizontal_pass(current, n, weight, fringe, surrogate)
            if vertical:
                current = current.transpose(-1, -2)
        return current

    def forward(self, value, surrogate=False):
        image = value.to(dtype=torch.float64)
        delta = self.lowpass(image, surrogate) - image
        threshold = self._quantize(self.tau.clamp(0, 255), torch.floor, surrogate)
        hard_gate = (delta.abs() > threshold).to(delta.dtype)
        if surrogate:
            soft_gate = torch.sigmoid(delta.abs() - threshold)
            gate = hard_gate + (soft_gate - soft_gate.detach())
        else:
            gate = hard_gate
        percent = self._quantize(self.gain.clamp(-4, 4) * 100, torch.round, surrogate)
        adjustment = self._quantize(delta * percent / 100, torch.trunc, surrogate)
        return (image + adjustment * gate).clamp(0, 255)

    @torch.no_grad()
    def predict(self, before):
        if before.device.type != "cpu":
            raise ValueError("IntegerLocalResponse prediction requires CPU input")
        byte = (before.to(dtype=torch.float32) * 255).round().to(dtype=torch.float64)
        return self(byte).to(dtype=before.dtype) / 255

    def parameter_record(self):
        """Complete JSON-ready parameters and conventions for typed hashing."""
        return dict(
            type=type(self).__module__ + "." + type(self).__qualname__,
            version="integer-local-response-v1", training=bool(self.training),
            state={name: dict(dtype=str(value.dtype), shape=list(value.shape),
                              values=value.detach().cpu().tolist())
                   for name, value in self.state_dict().items()},
            conventions=dict(
                Q=Q, rho_bounds=[0, RHO_MAX], gain_bounds=[-4, 4], tau_bounds=[0, 255],
                gain_percent="nearest-even round(gain*100), signed integer percent",
                threshold="floor(clamp(tau,0,255)); active iff abs(delta)>threshold",
                kernel="n=floor(rho); w=floor(Q/(2*rho+1)); f=floor((Q-(2*n+1)*w)/2); weights w on offsets -n..n and f at +/-(n+1)",
                passes=["horizontal"]*3 + ["vertical"]*3,
                boundary="replicate", accumulator="float64 exact integer forward via prefix sums",
                pass_quantization="floor((integer weighted sum + Q/2)/Q)",
                response="clip(x + trunc(delta*integer_percent/100)*gate,0,255)",
                input="float32 image*255, nearest-even round, cast float64",
                output="integer response cast to input dtype, then divide by 255",
                channels="same parameters independently shared by all three RGB channels",
            ),
        )

    @property
    def fingerprint(self):
        encoded = json.dumps(self.parameter_record(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def fit_support(cls, before, after, mask, device, check_budget=None):
        """Fit on one support content and its supplied pixel mask only."""
        device = torch.device(device)
        if device.type != "cpu":
            raise ValueError("The frozen single-support fit requires CPU")
        started = time.perf_counter()
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        seed_records, checkpoints = [], []

        def checkpoint(seed_index, completed_steps):
            if check_budget is not None:
                check_budget()
                checkpoints.append(dict(seed_index=seed_index, completed_steps=completed_steps))

        def snapshot(model):
            return {name: value.detach().clone() for name, value in model.state_dict().items()}

        try:
            # Normal tensors must be created outside an inherited inference mode.
            with torch.inference_mode(False), torch.enable_grad():
                x = torch.as_tensor(before, device=device, dtype=torch.float32).mul(255).round().to(torch.float64)
                y = torch.as_tensor(after, device=device, dtype=torch.float32).mul(255).round().to(torch.float64)
                fitmask = torch.as_tensor(mask, device=device, dtype=torch.bool).clone()
                if x.ndim != 3 or x.shape[0] != 3 or y.shape != x.shape or fitmask.shape != x.shape:
                    raise ValueError("Expected matching CHW RGB images and a complete boolean pixel mask")
                if not bool(fitmask.any()) or not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                    raise ValueError("Fit requires finite images and a nonempty training mask")
                best_model, best_seed, best_loss = None, None, float("inf")
                for seed_index, rho in enumerate(START_RADII):
                    checkpoint(seed_index, 0)
                    model = cls(device, rho=rho)
                    with torch.no_grad():
                        delta = model.lowpass(x) - x
                        denominator = delta[fitmask].square().sum().clamp_min(1e-12)
                        model.gain.copy_(((delta[fitmask] * (y-x)[fitmask]).sum()/denominator).clamp(-4, 4))
                        initial_gain = float(model.gain)
                    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
                    seed_best_loss, seed_best_state, seed_best_iteration = float("inf"), None, None
                    initial_loss = None
                    # The STE forward is exactly the integer forward. The loss
                    # before update t+1 scores iterate t, saving a second replay.
                    for step in range(1, STEPS+1):
                        optimizer.param_groups[0]["lr"] = INITIAL_LR * (STEPS-step+1) / STEPS
                        optimizer.zero_grad(set_to_none=True)
                        prediction = model(x, surrogate=True)
                        loss = (prediction[fitmask] - y[fitmask]).square().mean()
                        value = float(loss.detach())
                        if not bool(torch.isfinite(loss)):
                            raise ValueError("Non-finite integer-response training loss")
                        if initial_loss is None:
                            initial_loss = value
                        if value < seed_best_loss:
                            seed_best_loss, seed_best_state, seed_best_iteration = value, snapshot(model), step-1
                        loss.backward()
                        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                            raise ValueError("Missing or non-finite integer-response training gradient")
                        optimizer.step()
                        with torch.no_grad():
                            model.rho.clamp_(0, RHO_MAX)
                            model.gain.clamp_(-4, 4)
                            model.tau.clamp_(0, 255)
                        if step % 8 == 0:
                            checkpoint(seed_index, step)
                    with torch.no_grad():
                        terminal_loss = float((model(x)[fitmask]-y[fitmask]).square().mean())
                        if not torch.isfinite(torch.tensor(terminal_loss)):
                            raise ValueError("Non-finite terminal training loss")
                        if terminal_loss < seed_best_loss:
                            seed_best_loss, seed_best_state, seed_best_iteration = terminal_loss, snapshot(model), STEPS
                        model.load_state_dict(seed_best_state)
                    model.eval().requires_grad_(False)
                    seed_records.append(dict(
                        seed_index=seed_index, initial_rho=rho, initial_gain=initial_gain, initial_tau=0.0,
                        steps=STEPS, initial_train_mse8=initial_loss, terminal_train_mse8=terminal_loss,
                        best_train_mse8=seed_best_loss, best_iteration=seed_best_iteration,
                        parameter_record=model.parameter_record(), parameter_hash=model.fingerprint,
                    ))
                    if seed_best_loss < best_loss:
                        best_model, best_seed, best_loss = model, seed_index, seed_best_loss
                diagnostic = dict(
                    version="integer-local-response-single-support-fit-v1", seeds=seed_records,
                    selected_seed=best_seed, selected_train_mse8=best_loss,
                    parameter_record=best_model.parameter_record(), parameter_hash=best_model.fingerprint,
                    training_mask_pixels=int(fitmask.sum()), total_pixels=int(fitmask.numel()),
                    optimizer="Adam", steps_per_seed=STEPS, initial_lr=INITIAL_LR, final_lr=INITIAL_LR/STEPS,
                    lr_schedule="step t in 1..128 uses .03*(129-t)/128",
                    iterate_selection="Minimum exact integer-forward MSE8 on the supplied training mask among iterates 0..128; exact ties keep the earliest iterate",
                    seed_selection="Minimum training-mask MSE8; exact ties keep rho=.125 before rho=.5",
                    fitting_contents="This call's single anonymous before/after pair only",
                    budget_checkpoints=checkpoints, budget_callback_count=len(checkpoints),
                    device="cpu", threads=4, seconds=time.perf_counter()-started,
                )
                return best_model, diagnostic
        finally:
            torch.set_num_threads(previous_threads)
