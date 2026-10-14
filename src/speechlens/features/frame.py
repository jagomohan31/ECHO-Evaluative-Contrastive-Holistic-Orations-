from __future__ import annotations
import numpy as np
from speechlens.config import SR, HOP_S, FRAME_S

def _frames(samples, sr=SR, hop_s=HOP_S, frame_s=FRAME_S):
    hop = max(1, int(hop_s * sr)); win = max(1, int(frame_s * sr))
    return hop, win, max(1, (len(samples) - win) // hop + 1)

def energy_db(samples):
    hop, win, n = _frames(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        c = samples[i*hop: i*hop+win]
        out[i] = 20 * np.log10(np.sqrt(np.mean(c**2)) + 1e-8)
    return out

def energy_rel(energy, voiced_mask=None):
    src = energy[voiced_mask] if voiced_mask is not None and voiced_mask.any() else energy
    return energy - np.percentile(src, 90)

def f0_two_pass(samples):
    try:
        import parselmouth
        snd = parselmouth.Sound(samples, sampling_frequency=SR)
        p1 = snd.to_pitch_ac(time_step=HOP_S, pitch_floor=75.0, pitch_ceiling=600.0)
        v1 = p1.selected_array['frequency']; voiced1 = v1[v1 > 0]
        if len(voiced1) < 4:
            return np.zeros(len(samples)//max(1,int(HOP_S*SR))+1, dtype=np.float32)
        q1, q3 = np.percentile(voiced1, [25, 75])
        p2 = snd.to_pitch_ac(time_step=HOP_S, pitch_floor=max(50.,0.75*q1), pitch_ceiling=min(800.,1.5*q3))
        f0 = p2.selected_array['frequency'].astype(np.float32)
        for i in np.where(f0>0)[0]:
            lo=max(0,i-2); hi=min(len(f0),i+3)
            nb=f0[lo:hi][f0[lo:hi]>0]
            med=np.median(nb) if len(nb)>0 else 0
            if med>0 and abs(12*np.log2(f0[i]/med))>6: f0[i]=0.
        return f0
    except Exception:
        return np.zeros(len(samples)//max(1,int(HOP_S*SR))+1, dtype=np.float32)

def f0_semitones(f0_hz):
    voiced = f0_hz[f0_hz > 0]
    if len(voiced) == 0:
        return np.full(len(f0_hz), np.nan, dtype=np.float32)
    med = np.median(voiced)
    return np.where(f0_hz>0, 12*np.log2(np.where(f0_hz>0, f0_hz/med, 1.0)), np.nan).astype(np.float32)

def hnr(samples):
    try:
        import parselmouth
        snd = parselmouth.Sound(samples, sampling_frequency=SR)
        h = snd.to_harmonicity_cc(time_step=HOP_S, minimum_pitch=75.0)
        v = h.values[0].astype(np.float32); v[v<=-199]=np.nan; return v
    except Exception:
        return np.full(len(samples)//max(1,int(HOP_S*SR))+1, np.nan, dtype=np.float32)

def hnr_rel(hnr_arr):
    fin = hnr_arr[np.isfinite(hnr_arr)]
    ref = np.percentile(fin, 75) if len(fin)>0 else 0.
    return hnr_arr - ref

def mfcc_normalised(samples, n_mfcc=13):
    import librosa
    hop=int(HOP_S*SR); win=int(FRAME_S*SR)
    m=librosa.feature.mfcc(y=samples.astype(np.float32),sr=SR,n_mfcc=n_mfcc,n_fft=512,hop_length=hop,win_length=win)
    d=librosa.feature.delta(m); feat=np.vstack([m,d])
    mu=feat.mean(axis=1,keepdims=True); sd=feat.std(axis=1,keepdims=True)+1e-8
    return ((feat-mu)/sd).astype(np.float32)

def spectral_features(samples):
    import librosa
    hop=int(HOP_S*SR); win=int(FRAME_S*SR); y=samples.astype(np.float32)
    centroid=librosa.feature.spectral_centroid(y=y,sr=SR,n_fft=512,hop_length=hop,win_length=win)[0]
    flatness=librosa.feature.spectral_flatness(y=y,n_fft=512,hop_length=hop,win_length=win)[0]
    S=np.abs(librosa.stft(y,n_fft=512,hop_length=hop,win_length=win))**2
    freqs=librosa.fft_frequencies(sr=SR,n_fft=512)
    lo=(freqs>=100)&(freqs<=1000); hi=(freqs>=2000)&(freqs<=5000)
    e_lo=S[lo].sum(0)+1e-12; e_hi=S[hi].sum(0)+1e-12
    tilt_db=(10*np.log10(e_hi/e_lo)).astype(np.float32)
    ref=np.percentile(tilt_db[np.isfinite(tilt_db)],75) if tilt_db.size>0 else 0.
    return {'centroid':centroid.astype(np.float32),'flatness':flatness.astype(np.float32),'tilt_db':tilt_db,'tilt_rel':(tilt_db-ref).astype(np.float32)}

def clip_fraction(samples, win_s=0.025):
    hop=int(HOP_S*SR); win=int(win_s*SR); n=(len(samples)-win)//hop+1
    out=np.empty(n,dtype=np.float32)
    for i in range(n): out[i]=np.mean(np.abs(samples[i*hop:i*hop+win])>=0.99)
    return out

def modulation_spectrum_syllable_rate(samples):
    import librosa; y=samples.astype(np.float32)
    envelope=np.abs(librosa.effects.harmonic(y))+1e-8
    win_len=int(4.*SR); hop_win=int(1.*SR); peaks=[]
    for s in range(0,max(1,len(envelope)-win_len),hop_win):
        chunk=envelope[s:s+win_len]*np.hanning(min(win_len,len(envelope)-s))
        if len(chunk)<64: continue
        spec=np.abs(np.fft.rfft(chunk)); freqs=np.fft.rfftfreq(len(chunk),d=1./SR)
        mask=(freqs>=2)&(freqs<=8)
        if mask.any(): peaks.append(freqs[mask][np.argmax(spec[mask])])
    return float(np.median(peaks)) if peaks else 4.0

def resample_to_grid(series, src_hop_s, tgt_hop_s=HOP_S, n_frames=None):
    if n_frames is None: n_frames=int(len(series)*src_hop_s/tgt_hop_s)
    t_src=np.arange(len(series))*src_hop_s; t_tgt=np.arange(n_frames)*tgt_hop_s
    return np.interp(t_tgt,t_src,series).astype(np.float32)


# ---------------------------------------------------------------------------
# FrameFeatures bundle (used by word.py and downstream modules)
# ---------------------------------------------------------------------------
from dataclasses import dataclass as _dc, field as _field
from typing import Optional as _Opt

@_dc
class FrameFeatures:
    """All frame-level feature arrays for one recording, on the HOP_S grid."""
    times: "np.ndarray"                    # shape (n_frames,)
    energy_db: "np.ndarray"
    en_rel: "np.ndarray"
    f0_hz: "np.ndarray"
    f0_st: "np.ndarray"                    # semitones; NaN = unvoiced
    hnr_raw: "np.ndarray"
    hnr_rel: "np.ndarray"
    tilt_db: "np.ndarray"
    tilt_rel: "np.ndarray"
    clip_frac: "np.ndarray"
    syl_rate_hz: float = 4.0
    mfcc: "_Opt[np.ndarray]" = None        # shape (26, n_frames); None if skipped


def compute_frame_features(samples: "np.ndarray", sr: int = SR) -> FrameFeatures:
    """Compute all frame features for one recording and return a FrameFeatures bundle."""
    import warnings
    # Resample if needed
    if sr != SR:
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=SR)

    samples = samples.astype(np.float32)
    hop = int(HOP_S * SR)
    hop_s = HOP_S

    en = energy_db(samples)
    n = len(en)
    times = np.arange(n) * hop_s

    f0_hz_arr = f0_two_pass(samples)
    # Align lengths (parselmouth may return slightly different n)
    if len(f0_hz_arr) != n:
        t_src = np.arange(len(f0_hz_arr)) * hop_s
        f0_hz_arr = np.interp(times, t_src, f0_hz_arr).astype(np.float32)

    f0_st_arr = f0_semitones(f0_hz_arr)

    voiced_mask = f0_hz_arr > 0
    en_rel_arr = energy_rel(en, voiced_mask if voiced_mask.any() else None)

    h_raw = hnr(samples)
    if len(h_raw) != n:
        t_src = np.arange(len(h_raw)) * hop_s
        h_raw = np.interp(times, t_src, h_raw)
        h_raw = h_raw.astype(np.float32)
    h_rel = hnr_rel(h_raw)

    spec = spectral_features(samples)
    tilt = spec["tilt_db"]
    tilt_r = spec["tilt_rel"]
    if len(tilt) != n:
        t_src = np.arange(len(tilt)) * hop_s
        tilt = np.interp(times, t_src, tilt).astype(np.float32)
        tilt_r = np.interp(times, t_src, tilt_r).astype(np.float32)

    cf = clip_fraction(samples)
    if len(cf) != n:
        t_src = np.arange(len(cf)) * hop_s
        cf = np.interp(times, t_src, cf).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        syl_rate = modulation_spectrum_syllable_rate(samples)

    return FrameFeatures(
        times=times,
        energy_db=en,
        en_rel=en_rel_arr,
        f0_hz=f0_hz_arr,
        f0_st=f0_st_arr,
        hnr_raw=h_raw,
        hnr_rel=h_rel,
        tilt_db=tilt,
        tilt_rel=tilt_r,
        clip_frac=cf,
        syl_rate_hz=syl_rate,
        mfcc=None,
    )