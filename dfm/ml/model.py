"""Нейросеть B1 (документ B1, глава 8; шаг 4.3): свёрточная ветвь по растровому фрагменту кармана,
ветвь дескрипторов, учёт профиля процесса через FiLM, выходы — три квантиля относительного сужения
y и логит вероятности y > τ.

    фрагмент 3×64×64 ─ 4 блока «свёртка 3×3 — нормализация — FiLM(z) — ReLU», каналы 16/32/64/128,
                       размер вдвое меньше в каждом блоке ─ среднее по карте → 128
    дескрипторы     ─ 2 полносвязных слоя → 32
    [128 + 32] ─ FiLM(z) ─ 64 ─ 4 числа: ŷ₀,₅, ŷ₀,₅ − ŷ₀,₁ ≥ 0, ŷ₀,₉ − ŷ₀,₅ ≥ 0, логит

Профиль процесса z (ln Da, δ, t, EF, u, u_r) входит ТОЛЬКО через FiLM: F'_c = γ_c(z)·F_c + β_c(z)
(γ = 1 + Δγ — в начале обучения модуляция тождественна). В голове FiLM применяется и к признакам
дескрипторов: иначе зависимость «эффективный угол × промывка» шла бы только через свёрточную ветвь.
Квантили не пересекаются по построению (неотрицательные приращения через softplus).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

QUANTILES = (0.1, 0.5, 0.9)
CHANNELS = (16, 32, 64, 128)


class FiLMGen(nn.Module):
    """z → (γ, β) для всех модулируемых слоёв сразу."""

    def __init__(self, z_dim, sizes, hidden=64):
        super().__init__()
        self.sizes = list(sizes)
        self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2 * sum(self.sizes)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        out = self.net(z)
        res, i = [], 0
        for c in self.sizes:
            res.append((1 + out[:, i:i + c], out[:, i + c:i + 2 * c]))
            i += 2 * c
        return res


VARIANTS = ('full', 'no_raster', 'no_desc', 'no_foreign', 'z_input', 'no_align', 'ab_head', 'res32')
# абляции (документ B1, раздел 9.5): no_raster — только дескрипторы; no_desc — только растр; z_input —
# профиль обычным входом головы вместо FiLM; ab_head — выход ŷ = â + u_r·b̂ (u_r не входит в сеть).
# no_foreign, no_align, res32 меняют только вход (dfm/ml/train.py: prepare) — сеть та же, что full.


class B1Net(nn.Module):
    def __init__(self, n_desc, z_dim=6, in_ch=3, channels=CHANNELS, desc_hidden=32, head_hidden=64, variant='full',
                 hybrid=False):
        super().__init__()
        assert variant in VARIANTS, variant
        self.variant = variant
        self.hybrid = bool(hybrid)                       # ŷ = ŷ_формулы + поправка сети
        self.n_f = int(hybrid)                           # оценок формулы на входе: 1 — по θ_эфф, 2 — и по θ
        n_desc = n_desc + self.n_f                       # оценки формулы — ещё дескрипторы
        self.raster = variant != 'no_raster'
        self.use_desc = variant != 'no_desc'
        self.film_on = variant != 'z_input'
        self.ab = variant == 'ab_head'
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        c0 = in_ch
        for c in (channels if self.raster else ()):
            self.convs.append(nn.Conv2d(c0, c, 3, padding=1, bias=False))
            self.norms.append(nn.BatchNorm2d(c))
            c0 = c
        self.desc = (nn.Sequential(nn.Linear(n_desc, desc_hidden), nn.ReLU(), nn.Linear(desc_hidden, desc_hidden),
                                   nn.ReLU()) if self.use_desc else None)
        feat = (channels[-1] if self.raster else 0) + (desc_hidden if self.use_desc else 0)
        zf = z_dim - 1 if self.ab else z_dim            # в ab_head u_r в сеть не подаётся
        if self.film_on:
            self.film = FiLMGen(zf, (list(channels) if self.raster else []) + [feat])
        else:
            self.zemb = nn.Sequential(nn.Linear(zf, 32), nn.ReLU())
            feat += 32
        self.head = nn.Sequential(nn.Linear(feat + self.n_f, head_hidden), nn.ReLU(),
                                  nn.Linear(head_hidden, 8 if self.ab else 4))
        if hybrid:                                       # в начале обучения ŷ₀,₅ = ŷ_формулы
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, d, z, u_r=None, f=None):
        """x — фрагмент, d — дескрипторы, z — нормированный профиль (последний столбец — u_r),
        u_r — u_r в мкм (нужен варианту ab_head), f — сужение по формуле клина при этом u_r (гибрид):
        (N,) — по θ_эфф, (N, 2) — по θ_эфф и по углу вершины θ; основа выхода — по θ_эфф."""
        if self.hybrid:
            f = f[:, None] if f.ndim == 1 else f
            d = torch.cat([d, f], dim=1)
        if self.ab:
            z = z[:, :-1]
        mods = self.film(z) if self.film_on else [(None, None)] * (len(self.convs) + 1)
        parts_ = []
        if self.raster:
            for conv, norm, (g, b) in zip(self.convs, self.norms, mods):
                x = norm(conv(x))
                if g is not None:
                    x = g[:, :, None, None] * x + b[:, :, None, None]
                x = F.max_pool2d(F.relu(x), 2)
            parts_.append(x.mean(dim=(2, 3)))
        if self.use_desc:
            parts_.append(self.desc(d))
        h = torch.cat(parts_, dim=1)
        if self.film_on:
            g, b = mods[-1]
            h = g * h + b
        else:
            h = torch.cat([h, self.zemb(z)], dim=1)
        if self.hybrid:
            h = torch.cat([h, f], dim=1)
        o = self.head(h)
        base = f[:, 0] if self.hybrid else 0.0
        if self.ab:
            u = u_r                                      # (N,) мкм
            q50 = base + o[:, 0] + u * o[:, 3]
            q10 = q50 - F.softplus(o[:, 1]) - u * F.softplus(o[:, 4])
            q90 = q50 + F.softplus(o[:, 2]) + u * F.softplus(o[:, 5])
            return torch.stack([q10, q50, q90], dim=1), o[:, 6] + u * o[:, 7]
        q50 = base + o[:, 0]
        q10 = q50 - F.softplus(o[:, 1])
        q90 = q50 + F.softplus(o[:, 2])
        return torch.stack([q10, q50, q90], dim=1), o[:, 3]


def pinball(y, q_pred, quantiles=QUANTILES):
    """Потеря пинбола, сумма по квантилям, среднее по примерам: max(q·e, (q − 1)·e), e = y − ŷ."""
    q = torch.as_tensor(quantiles, dtype=q_pred.dtype, device=q_pred.device)
    e = y[:, None] - q_pred
    return torch.maximum(q * e, (q - 1) * e).sum(dim=1).mean()


def focal(logit, target, gamma=2.0, alpha=0.75):
    """Фокальная потеря (документ B2, раздел 8.7): (1 − p_t)^γ·BCE, α — вес редкого класса «опасно»."""
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    p_t = torch.exp(-bce)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return (a_t * (1 - p_t) ** gamma * bce).mean()


def n_params(model):
    return sum(p.numel() for p in model.parameters())
