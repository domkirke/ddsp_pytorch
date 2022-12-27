import torch
import torch.nn as nn
import torchaudio as ta
from .core import mlp, gru, scale_function, remove_above_nyquist, upsample
from .core import harmonic_synth, amp_to_impulse_response, fft_convolve
from .core import resample
import math


class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x



class DDSP(nn.Module):
    def __init__(self, 
                 latent_size = 16,
                 frequency_size = 1,
                 loudness_size = 1,
                 hidden_size = 512,
                 n_harmonic = 100,
                 n_bands = 65,
                 sampling_rate = 16000,
                 block_size = 160, 
                 reverb = True):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.latent_size = latent_size

        in_mlps = [mlp(frequency_size, hidden_size, 3)]*2
        if self.latent_size > 0:
            in_mlps.append(mlp(latent_size, hidden_size, 3))
        self.in_mlps = nn.ModuleList(in_mlps)

        self.gru = gru(2 + int(self.latent_size > 0), hidden_size)
        self.out_mlp = mlp((3 + int(self.latent_size > 0))*hidden_size, hidden_size, 3)

        # dynamic loudness regularization
        self.loudness_bn = nn.BatchNorm2d(loudness_size, momentum=0.0)
        # init batchnorm to fit -80-0dB range
        self.loudness_bn.running_mean = torch.tensor([-40.])
        self.loudness_bn.running_var = torch.tensor([40.])

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.has_reverb = reverb
        if self.has_reverb:
            self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, pitch, loudness, z = None):
        # hidden = torch.cat([
        #     self.in_mlps[0](pitch),
        #     self.in_mlps[1](loudness),
        # ], -1)
        # hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
        # hidden = self.out_mlp(hidden)
        loudness = self.loudness_bn(loudness.unsqueeze(-3))[..., 0, :, :]
        desc_inputs = [self.in_mlps[0](pitch), self.in_mlps[1](loudness)]
        if (self.latent_size > 0) and (z is not None):
            desc_inputs.append(self.in_mlps[2](z))
        desc_embeddings = torch.cat(desc_inputs, -1)
        hidden = torch.cat([self.gru(desc_embeddings)[0], desc_embeddings], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        #reverb part
        if self.has_reverb:
            signal = self.reverb(signal)

        return signal[..., 0]

    def realtime_forward(self, pitch, loudness):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)

        gru_out, cache = self.gru(hidden, self.cache_gru)
        self.cache_gru.copy_(cache)

        hidden = torch.cat([gru_out, pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / self.sampling_rate, 1)

        omega = omega + self.phase
        self.phase.copy_(omega[0, -1, 0] % (2 * math.pi))

        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)

        harmonic = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        return signal