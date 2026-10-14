"""tests/test_t05_frame_features.py - synthetic signal tests, no model."""
import numpy as np, pytest
SR = 16000

def _sine(freq, dur=1.0):
    t = np.linspace(0, dur, int(dur*SR), endpoint=False)
    return (np.sin(2*np.pi*freq*t) * 0.5).astype(np.float32)

def test_energy_half_amplitude():
    from speechlens.features.frame import energy_db
    s1 = _sine(440, 0.5); s2 = s1 * 0.5
    e1 = energy_db(s1).mean(); e2 = energy_db(s2).mean()
    assert abs((e2 - e1) - (-6.02)) < 0.5

def test_f0_220hz():
    from speechlens.features.frame import f0_two_pass
    s = _sine(220, 1.0)
    f0 = f0_two_pass(s)
    voiced = f0[f0 > 0]
    if len(voiced) > 0:
        assert abs(np.median(voiced) - 220) < 10

def test_f0_semitones_self_zero():
    from speechlens.features.frame import f0_semitones
    f0 = np.array([220.0]*50, dtype=np.float32)
    st = f0_semitones(f0)
    assert np.allclose(st[np.isfinite(st)], 0.0, atol=0.1)

def test_f0_semitones_relative():
    """Semitones are relative to speaker median. With 220 Hz median, 440 Hz = +12 st."""
    from speechlens.features.frame import f0_semitones
    import numpy as np
    # All frames at 220 Hz (median=220), except last 10 at 440 Hz
    f0 = np.array([220.0]*50 + [440.0]*10, dtype=np.float32)
    st = f0_semitones(f0)
    upper = st[50:][np.isfinite(st[50:])]
    if len(upper) > 0:
        assert abs(np.median(upper) - 12.0) < 1.0

def test_clip_fraction_clean():
    from speechlens.features.frame import clip_fraction
    s = _sine(440, 0.5)
    cf = clip_fraction(s)
    assert cf.mean() < 0.01

def test_clip_fraction_clipped():
    from speechlens.features.frame import clip_fraction
    s = np.clip(_sine(440, 0.5) * 10, -1.0, 1.0)
    cf = clip_fraction(s)
    assert cf.mean() > 0.1

def test_resample_grid():
    from speechlens.features.frame import resample_to_grid
    src = np.arange(50, dtype=np.float32)
    out = resample_to_grid(src, src_hop_s=0.02, tgt_hop_s=0.01, n_frames=100)
    assert len(out) == 100
    assert out[0] == pytest.approx(0.0, abs=0.1)
