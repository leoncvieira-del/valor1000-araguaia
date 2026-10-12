"""Trilha original sintetizada em código (120 BPM) + efeitos de interface.

Uso: python3 trilha.py eventos.json saida.wav
Os eventos (cliques, whooshes, impactos) vêm da própria linha do tempo (index.html -> window.EVENTS).
Grade musical: compasso começa em 3,0 s (o gancho de 0–3 s é anacruse de 6 tempos).
"""
import json
import sys
import wave

import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR)
BEAT = 0.5
rng = np.random.default_rng(1978)

ev = json.load(open(sys.argv[1]))
OUT = sys.argv[2]

PAD = SR * 4
bus = {k: np.zeros((2, N + PAD)) for k in ('drums', 'bass', 'music', 'sfx', 'verb', 'delay')}


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tt(n):
    return np.arange(n) / SR


def add(name, x, t, pan=0.0, gain=1.0):
    i = int(round(t * SR))
    if x.ndim == 1:
        a = np.pi / 4 * (pan + 1)
        x = np.stack([x * np.cos(a), x * np.sin(a)]) * np.sqrt(2)
    if i < 0:
        x = x[:, -i:]
        i = 0
    n = min(x.shape[1], bus[name].shape[1] - i)
    if n > 0:
        bus[name][:, i:i + n] += x[:, :n] * gain


def spectral(x, fn):
    """Filtra um sinal mono pelo ganho fn(freqs) no domínio da frequência."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(X * fn(f), len(x))


def bandnoise(dur, lo, hi):
    x = rng.standard_normal(int(dur * SR))
    return spectral(x, lambda f: ((f > lo) & (f < hi)).astype(float))


def sweep_noise(dur, f0, f1, q=0.9):
    """Ruído com banda passante cujo centro varre de f0 a f1 (STFT)."""
    n = int(dur * SR)
    hop, win = 256, 1024
    x = rng.standard_normal(n + win)
    y = np.zeros(n + win)
    w = np.hanning(win)
    f = np.fft.rfftfreq(win, 1 / SR)
    for s in range(0, n, hop):
        p = s / max(n - 1, 1)
        fc = f0 * (f1 / f0) ** p
        g = np.exp(-0.5 * (np.log2(np.maximum(f, 1) / fc) / q) ** 2)
        y[s:s + win] += np.fft.irfft(np.fft.rfft(x[s:s + win] * w) * g, win) * w
    return y[:n] / 1.5


def env_adsr(n, a=0.005, d=0.2, s=0.0, r=0.05, hold=None):
    t = tt(n)
    e = np.minimum(t / max(a, 1e-4), 1.0)
    e = e * (s + (1 - s) * np.exp(-np.maximum(t - a, 0) / max(d, 1e-4)))
    if hold is not None:
        e *= np.clip(1 - (t - hold) / r, 0, 1)
    return e


def saw(freq, dur, cutoff=7000, decay_k=None, phase=0.0):
    """Dente de serra aditivo; decay_k faz os harmônicos agudos morrerem antes (filtro com envelope)."""
    n = int(dur * SR)
    t = tt(n)
    K = max(1, int(min(cutoff, SR / 2 - 2000) / freq))
    x = np.zeros(n)
    for k in range(1, K + 1):
        amp = 1.0 / k
        h = np.sin(2 * np.pi * k * freq * t + phase * k)
        if decay_k is not None:
            h *= np.exp(-t * decay_k * (k ** 0.8))
        x += amp * h
    return x


# ---------------- instrumentos ----------------
def kick(gain=1.0, length=0.45):
    n = int(length * SR)
    t = tt(n)
    f = 46 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-t * 6.5)
    x += 0.35 * bandnoise(length, 1500, 6000) * np.exp(-t * 180)
    return np.tanh(x * 1.6) * gain


def clap():
    n = int(0.35 * SR)
    t = tt(n)
    e = np.zeros(n)
    for o in (0, 0.011, 0.022):
        e += (t >= o) * np.exp(-np.maximum(t - o, 0) * (60 if o < 0.02 else 14))
    return bandnoise(0.35, 900, 5200) * e * 0.55


def hat(open_=False):
    d = 0.16 if open_ else 0.04
    t = tt(int(d * SR))
    return bandnoise(d, 7000, 16000) * np.exp(-t * (22 if open_ else 110)) * (0.35 if open_ else 0.28)


def tick():
    t = tt(int(0.05 * SR))
    return (np.sin(2 * np.pi * 1850 * t) * 0.6 + bandnoise(0.05, 3000, 9000) * 0.3) * np.exp(-t * 90)


def sub_hit(gain=1.0):
    t = tt(int(1.4 * SR))
    f = 34 + 70 * np.exp(-t * 5)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.4)
    x += bandnoise(1.4, 60, 900) * np.exp(-t * 9) * 0.5
    return np.tanh(x * 1.4) * gain


def bass_note(m, dur=0.22):
    x = saw(mtof(m), dur, cutoff=3000, decay_k=7)
    x += 0.6 * np.sin(2 * np.pi * mtof(m) * tt(len(x)))
    return x * env_adsr(len(x), a=0.003, d=0.18, s=0.25, r=0.03, hold=dur - 0.03) * 0.5


def supersaw(notes, dur, cutoff=5200, voices=(-14, -6, 0, 6, 14), decay=0.28, sustain=0.0):
    n = int(dur * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    for m in notes:
        for j, c in enumerate(voices):
            f = mtof(m) * 2 ** (c / 1200)
            x = saw(f, dur, cutoff=cutoff, phase=rng.uniform(0, 6.28))
            pan = (j / (len(voices) - 1)) * 2 - 1
            a = np.pi / 4 * (pan * 0.8 + 1)
            L += x * np.cos(a)
            R += x * np.sin(a)
    e = env_adsr(n, a=0.004, d=decay, s=sustain, r=0.06, hold=dur - 0.06)
    return np.stack([L, R]) * e / (len(notes) * len(voices)) * 1.6


def pluck(m, dur=0.22, bright=1.0):
    f = mtof(m)
    t = tt(int(dur * SR))
    x = np.zeros(len(t))
    for k in (1, 3, 5, 7, 9):
        x += np.sin(2 * np.pi * k * f * t) / k * np.exp(-t * (14 + 9 * k / bright))
    return x * np.minimum(t / 0.002, 1) * 0.5


def bell(m, dur=2.5):
    f = mtof(m)
    t = tt(int(dur * SR))
    mod = np.sin(2 * np.pi * f * 3.5 * t) * 2.2 * np.exp(-t * 3)
    return np.sin(2 * np.pi * f * t + mod) * np.exp(-t * 1.8) * np.minimum(t / 0.002, 1) * 0.35


def pad(notes, dur, attack=0.6):
    t = tt(int(dur * SR))
    x = np.zeros(len(t))
    for m in notes:
        for c in (-8, 8):
            f = mtof(m) * 2 ** (c / 1200)
            x += np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
    e = np.minimum(t / attack, 1) * np.clip((dur - t) / 0.4, 0, 1)
    return x * e / (2 * len(notes)) * 0.5


# ---------------- harmonia ----------------
CH = {
    'Dm': ([62, 65, 69, 74], 38), 'Bb': ([62, 65, 70, 74], 34),
    'F': ([60, 65, 69, 72], 41), 'C': ([60, 64, 67, 72], 36),
}
PROG = ['Dm', 'Bb', 'F', 'C', 'Dm', 'Bb', 'C']   # compassos a partir de 3,0 s


def chord_at(t):
    i = int((t - 3.0) // 2)
    return CH[PROG[max(0, min(i, len(PROG) - 1))]]


# ---------------- arranjo ----------------
# Gancho 0–3 s: tique-taque de urgência + impactos em cada palavra + riser
for i in range(12):
    tk = i * 0.25
    if tk < 2.0:
        add('drums', tick(), tk, pan=0.3 if i % 2 else -0.3, gain=0.35 if i % 2 else 0.5)
add('music', pad([50, 57, 62], 3.2, attack=1.5), 0.0, gain=0.9)
for tm, g in ev['impacts']:
    if tm < 2.1:
        add('drums', kick(1.0), tm, gain=0.95 * g)
        add('drums', sub_hit(0.6 * g), tm)
        add('verb', bandnoise(0.3, 200, 3000) * np.exp(-tt(int(0.3 * SR)) * 20) * 0.6 * g, tm)
# riser 2,0–2,96
r = sweep_noise(0.96, 300, 9000, q=0.7) * np.linspace(0.1, 1.0, int(0.96 * SR)) ** 2
add('sfx', r, 2.0, gain=0.55)
tr = tt(int(0.96 * SR))
fr = 110 * 2 ** (tr * 3.2)
add('music', np.sin(2 * np.pi * np.cumsum(fr) / SR) * (tr / 0.96) ** 2 * 0.22, 2.0)
# caixa acelerando 2,5–2,94
roll = list(np.arange(2.5, 2.75, 0.0625)) + list(np.arange(2.75, 2.94, 0.03125))
for k, tm in enumerate(roll):
    add('drums', clap(), tm, gain=0.25 + 0.6 * k / len(roll))

# Drop em 3,0 s
add('drums', sub_hit(1.0), 3.0)
add('verb', clap() * 1.5, 3.0)
GROOVE = [(3.0, 13.5), (15.0, 16.0)]


def in_groove(tm):
    return any(a <= tm < b - 1e-6 for a, b in GROOVE)


kicks = [round(3.0 + j * BEAT, 3) for j in range(27)] + [15.0, 15.5]
kicks = sorted(set(k for k in kicks if in_groove(k)))
for k in kicks:
    add('drums', kick(), k, gain=0.95)
for j in range(int((16.5 - 3.0) / 0.125)):
    tm = 3.0 + j * 0.125
    if not in_groove(tm):
        continue
    pos = j % 4
    if pos == 2:
        add('drums', hat(True), tm, pan=0.25)
    else:
        add('drums', hat(), tm, pan=-0.2 if pos % 2 else 0.15, gain=1.0 if pos == 0 else 0.6)
    if j % 8 == 4:
        add('drums', clap(), tm, gain=0.9)
        add('verb', clap(), tm, gain=0.5)
# baixo em contratempo de colcheia + nota no tempo forte a cada compasso
for j in range(int((16.5 - 3.0) / 0.25)):
    tm = 3.0 + j * 0.25
    if not in_groove(tm):
        continue
    _, root = chord_at(tm)
    if j % 2 == 1:
        add('bass', bass_note(root + (12 if j % 8 == 7 else 0)), tm)
    elif j % 8 == 0:
        add('bass', bass_note(root, 0.2), tm, gain=0.8)
# acordes sincopados (posições em semicolcheias dentro do compasso de 2 s)
STAB = [0, 3, 6, 10, 12]
for bar in range(7):
    b0 = 3.0 + bar * 2
    for s16 in STAB:
        tm = b0 + s16 * 0.125
        if not in_groove(tm):
            continue
        notes, _ = chord_at(tm)
        g = 0.85 if b0 < 6 else 0.6
        x = supersaw(notes, 0.34, cutoff=4800 if b0 < 6 else 3800)
        add('music', x, tm, gain=g)
        add('verb', x, tm, gain=0.25)
# arpejo dos recursos (6,0–13,5 e 15,0–16,0)
ARP = [0, 1, 2, 3, 2, 1, 3, 2]
for j in range(int((16.5 - 6.0) / 0.125)):
    tm = 6.0 + j * 0.125
    if not (6.0 <= tm < 13.5 or 15.0 <= tm < 16.0):
        continue
    notes, _ = chord_at(tm)
    m = notes[ARP[j % 8]] + 12
    x = pluck(m, 0.2, bright=1.3)
    add('music', x, tm, pan=0.35 if j % 2 else -0.35, gain=0.45)
    add('delay', x, tm, gain=0.4)
# respiro do número 13,5–15,0: pad + contagem com blips ascendentes
add('music', pad([62, 65, 70], 1.6, attack=0.3), 13.5, gain=1.1)
c0, c1 = ev['count']
PENTA = [65, 67, 69, 72, 74, 77, 79, 81, 84, 86, 89, 91, 93, 96, 98]   # Fá pentatônica: uma nota a cada 2 folhas
for k in range(30):
    tm = c0 + (c1 - c0) * k / 30
    add('sfx', pluck(PENTA[k // 2], 0.08, bright=2), tm, pan=0.5 * np.sin(k * 2.4), gain=0.3)
# caminho até a assinatura: riser 16,0–16,5
r = sweep_noise(0.5, 500, 10000, q=0.6) * np.linspace(0, 1, int(0.5 * SR)) ** 1.5
add('sfx', r, 16.0, gain=0.5)
# assinatura 16,5–20: Fá maior grande, sino, letras em arpejo
FINAL = [53, 57, 60, 65, 69, 72, 77]
add('drums', kick(1.0, 0.6), 16.5)
add('drums', sub_hit(0.9), 16.5)
x = supersaw(FINAL, 3.4, cutoff=5500, decay=1.4, sustain=0.25)
add('music', x, 16.5, gain=0.85)
add('verb', x, 16.5, gain=0.45)
add('music', pad([41, 48, 53, 57, 60], 3.5, attack=0.05), 16.5, gain=1.2)
SCALE = [77, 79, 81, 84, 86, 89, 91, 93]
for i, tm in enumerate(ev['pops'][2:10]):
    p = pluck(SCALE[i], 0.35, bright=2.2)
    add('music', p, tm, pan=-0.6 + i * 0.17, gain=0.5)
    add('verb', p, tm, gain=0.4)
add('music', bell(96, 2.8), 16.95, gain=0.6)
add('verb', bell(89, 2.8), 16.95, gain=0.5)
add('music', bell(84, 2.0), 18.5, gain=0.35)
add('verb', bell(84, 2.0), 18.5, gain=0.4)

# ---------------- efeitos de interface ----------------
def click_sfx(soft=False):
    t = tt(int(0.06 * SR))
    x = bandnoise(0.06, 2000, 12000) * np.exp(-t * 600) * 0.9
    x += np.sin(2 * np.pi * (2600 if not soft else 1900) * t) * np.exp(-t * 140) * 0.45
    x += np.sin(2 * np.pi * 180 * t) * np.exp(-t * 70) * 0.5
    return x * (0.55 if soft else 1.0)


SOFT_UPS = set(ev.get('soft', []))
for c in ev['clicks']:
    add('sfx', click_sfx(c in SOFT_UPS), c, pan=0.15, gain=0.9)
    add('verb', click_sfx(), c, gain=0.12)
for a, b, kind in ev['whoosh']:
    d = b - a
    n = int(d * SR)
    shape = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5
    if kind == 'zoom':
        w, g = sweep_noise(d, 400, 7000, q=0.6) * np.linspace(0, 1, n) ** 2, 0.7
    elif kind == 'scroll':
        w, g = sweep_noise(d, 1200, 2500, q=0.8) * shape, 0.25
    elif kind == 's':
        w, g = sweep_noise(d, 900, 4500, q=0.7) * shape, 0.35
    else:
        w, g = sweep_noise(d, 250, 5000, q=0.8) * shape, 0.7
    # whoosh atravessa o estéreo
    pan = np.linspace(-0.8, 0.8, n)
    ang = np.pi / 4 * (pan + 1)
    add('sfx', np.stack([w * np.cos(ang), w * np.sin(ang)]) * np.sqrt(2), a, gain=g)
for tm, g in ev['impacts']:
    if tm >= 2.9:
        add('sfx', sub_hit(0.55 * g), tm)
add('sfx', pluck(84, 0.2, bright=2), ev['pops'][0], gain=0.45)        # QR "pop"
for i, tm in enumerate(ev['pops'][1:2]):
    add('sfx', bandnoise(0.12, 300, 3000) * np.exp(-tt(int(0.12 * SR)) * 30) * 0.5, tm)
for tm in ev.get('bag', []):                                         # sacaria entra
    add('sfx', sub_hit(0.35), tm)
    add('sfx', pluck(77, 0.25, bright=1.5), tm, gain=0.35)
for k, tm in enumerate(ev.get('chips', [])):                          # categorias saltam da sacaria
    p = pluck([77, 81, 84, 86, 89][k % 5], 0.25, bright=2.2)
    add('sfx', p, tm, pan=-0.6 + 0.3 * k, gain=0.42)
    add('verb', p, tm, gain=0.3)
for tm in ev['ticks']:
    t = tt(int(0.03 * SR))
    add('sfx', bandnoise(0.03, 2500, 9000) * np.exp(-t * 300) * rng.uniform(0.4, 0.6), tm + rng.uniform(-0.006, 0.006),
        pan=rng.uniform(-0.2, 0.2))

# ---------------- mixagem ----------------
def reverb(x, secs=2.2, decay=3.0, lp=6500):
    n = int(secs * SR)
    t = tt(n)
    ir = np.stack([spectral(rng.standard_normal(n), lambda f: 1 / (1 + (f / lp) ** 2)) * np.exp(-t * decay)
                   for _ in range(2)])
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    L = x.shape[1] + n
    F = 1 << (L - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, F) * np.fft.rfft(ir, F), F)[:, :x.shape[1]]
    return y


def pingpong(x, d=0.375, fb=0.38, taps=4):
    y = np.zeros_like(x)
    s = int(d * SR)
    for k in range(1, taps + 1):
        g = fb ** k
        src = x[0] + x[1]
        ch = k % 2
        y[ch, k * s:] += src[:-k * s] * g * 0.5
    return y


# sidechain (bombeia o resto da música no ritmo do bumbo)
side = np.ones(N + PAD)
tk = tt(int(0.3 * SR))
for k in kicks + [3.0, 16.5]:
    i = int(k * SR)
    seg = 1 - 0.55 * np.exp(-tk / 0.07)
    n = min(len(seg), len(side) - i)
    side[i:i + n] = np.minimum(side[i:i + n], seg[:n])

mix = (bus['drums'] * 0.9
       + bus['bass'] * side * 0.9
       + bus['music'] * side * 0.8
       + reverb(bus['verb']) * side * 0.5
       + pingpong(bus['delay']) * side * 0.6
       + bus['sfx'] * 0.85)
mix = mix[:, :N]
mix = np.stack([spectral(c, lambda f: 1 / np.sqrt(1 + (28 / np.maximum(f, 1)) ** 4)) for c in mix])
fade = np.ones(N)
fn = int(0.7 * SR)
fade[-fn:] = np.linspace(1, 0, fn) ** 1.5
mix *= fade
mix /= np.abs(mix).max()
mix = np.tanh(mix * 1.5) / np.tanh(1.5)
mix *= 10 ** (-1 / 20) / np.abs(mix).max()
pcm = (mix.T * 32767).astype(np.int16)
with wave.open(OUT, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('ok', OUT, f'{N / SR:.2f}s')
