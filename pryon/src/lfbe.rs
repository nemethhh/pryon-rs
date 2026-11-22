//! Streaming LFBE front end (AED 20 bins 100-6000 Hz, wake 64 bins 80-7200 Hz), bit-identical to the C reference.
//!
//! Third-party: `msun_logf` ports FreeBSD msun e_logf.c (Sun, permissive); the FFT operation order follows
//! Ne10 (BSD-3-Clause); the window/power/mel stages follow Kaldi (Apache-2.0). See THIRD_PARTY_NOTICES.md.
use crate::lfbe_tables::{NE_SU, NE_TW, POVEY};
use crate::num::sdot_vfp;

pub const BINS: usize = 20;
const FL: usize = 400;
const HOP: usize = 160;
const N: usize = 512;
const TW81: f32 = 0.70710678;
const TW81N: f32 = -0.70710678;

/// FreeBSD msun `__ieee754_logf`, including its non-finite and subnormal handling (no libm).
pub(crate) fn msun_logf(x: f32) -> f32 {
    const LN2_HI: f32 = 6.9313812256e-01;
    const LN2_LO: f32 = 9.0580006145e-06;
    const LG1: f32 = 0xaaaaaau32 as f32 / 16777216.0;
    const LG2: f32 = 0xccce13u32 as f32 / 33554432.0;
    const LG3: f32 = 0x91e9eeu32 as f32 / 33554432.0;
    const LG4: f32 = 0xf89e26u32 as f32 / 67108864.0;
    let mut x = x;
    let mut ix = x.to_bits() as i32;
    let mut k: i32 = 0;
    if ix < 0x0080_0000 {
        if ix & 0x7fff_ffff == 0 {
            return f32::NEG_INFINITY;
        }
        if ix < 0 {
            return f32::NAN;
        }
        k -= 25;
        x *= 33554432.0;
        ix = x.to_bits() as i32;
    }
    if ix >= 0x7f80_0000 {
        return x + x;
    }
    k += (ix >> 23) - 127;
    ix &= 0x007f_ffff;
    let i = (ix + (0x95f64 << 3)) & 0x80_0000;
    x = f32::from_bits((ix | (i ^ 0x3f80_0000)) as u32);
    k += i >> 23;
    let f = x - 1.0;
    if (0x007f_ffff & (0x8000 + ix)) < 0xc000 {
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            let dk = k as f32;
            return dk * LN2_HI + dk * LN2_LO;
        }
        let r = f * f * (0.5 - 0.33333333333333333f32 * f);
        if k == 0 {
            return f - r;
        }
        let dk = k as f32;
        return dk * LN2_HI - ((r - dk * LN2_LO) - f);
    }
    let s = f / (2.0 + f);
    let dk = k as f32;
    let z = s * s;
    let mut i = ix - (0x6147a << 3);
    let w = z * z;
    let j = (0x6b851 << 3) - ix;
    let t1 = w * (LG2 + w * LG4);
    let t2 = z * (LG1 + w * LG3);
    i |= j;
    let r = t2 + t1;
    if i > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            return f - (hfsq - s * (hfsq + r));
        }
        return dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f);
    }
    if k == 0 {
        return f - s * (f - r);
    }
    dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
}

// Ne10 ne10_fft_r2c_1d_float32_neon (nfft 512, unroll 1) in the same operation order, with [f32; 4] lanes.
type V = [f32; 4];
#[derive(Clone, Copy)]
struct C4 {
    r: V,
    i: V,
}

#[inline(always)]
fn ld(p: &[f32], o: usize) -> V {
    [p[o], p[o + 1], p[o + 2], p[o + 3]]
}
#[inline(always)]
fn st(p: &mut [f32], o: usize, v: V) {
    p[o..o + 4].copy_from_slice(&v);
}
#[inline(always)]
fn dup(x: f32) -> V {
    [x; 4]
}
#[inline(always)]
fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}
#[inline(always)]
fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}
#[inline(always)]
fn mul(a: V, b: V) -> V {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
}
#[inline(always)]
fn neg(a: V) -> V {
    [-a[0], -a[1], -a[2], -a[3]]
}
#[inline(always)]
fn rev4(a: V) -> V {
    [a[3], a[2], a[1], a[0]]
}
#[inline(always)]
fn cadd(a: C4, b: C4) -> C4 {
    C4 { r: add(a.r, b.r), i: add(a.i, b.i) }
}
#[inline(always)]
fn csub(a: C4, b: C4) -> C4 {
    C4 { r: sub(a.r, b.r), i: sub(a.i, b.i) }
}
#[inline(always)]
fn cmul(a: C4, b: C4) -> C4 {
    let (rr, ri, p, q) = (mul(a.r, b.r), mul(a.r, b.i), mul(a.i, b.i), mul(a.i, b.r));
    C4 { r: sub(rr, p), i: add(ri, q) }
}

fn radix8x4(out: &mut [f32], inp: &[f32]) {
    for f in 0..16 {
        let mut q = [[0f32; 4]; 8];
        for (k, qk) in q.iter_mut().enumerate() {
            *qk = ld(inp, 4 * (f + 16 * k));
        }
        let mut s = [
            add(q[0], q[4]),
            sub(q[0], q[4]),
            add(q[1], q[5]),
            sub(q[1], q[5]),
            add(q[2], q[6]),
            sub(q[2], q[6]),
            add(q[3], q[7]),
            sub(q[3], q[7]),
        ];
        s[3] = mul(s[3], dup(TW81));
        s[7] = mul(s[7], dup(TW81N));
        let s0 = add(s[0], s[4]);
        let s1 = add(s[2], s[6]);
        let s2 = sub(s[7], s[3]);
        let s3 = add(s[3], s[7]);
        let o = [
            add(s0, s1),
            add(s[1], s3),
            sub(s2, s[5]),
            sub(s[0], s[4]),
            sub(s[6], s[2]),
            sub(s[1], s3),
            add(s[5], s2),
            sub(s0, s1),
        ];
        for (k, ok) in o.iter().enumerate() {
            st(out, 4 * (8 * f + k), *ok);
        }
    }
}

fn radix4x4_tw(out: &mut [f32], inp: &[f32], fstride: usize, ms: usize, tw: &[[f32; 2]]) {
    const IS: usize = 32;
    let (mut fin, mut fo) = (0usize, 0usize);
    for _ in 0..fstride {
        {
            let q0 = ld(inp, fin);
            let q1 = ld(inp, fin + 4 * IS);
            let q2 = ld(inp, fin + 8 * IS);
            let q3 = ld(inp, fin + 12 * IS);
            let s0 = add(q0, q2);
            let s1 = add(q1, q3);
            st(out, fo, add(s0, s1));
            st(out, fo + 4 * (2 * ms - 1), sub(q0, q2));
            st(out, fo + 4 * (2 * ms), sub(q3, q1));
            st(out, fo + 4 * (4 * ms - 1), sub(s0, s1));
        }
        fin += 4;
        fo += 4;
        {
            let (mut fi, mut ti, mut fr, mut fb) = (fin, 3usize, fo, fo + 4 * (4 * ms - 4));
            for _ in 0..ms / 2 - 1 {
                let mut inn = [C4 { r: [0.; 4], i: [0.; 4] }; 4];
                for (k, c) in inn.iter_mut().enumerate() {
                    c.r = ld(inp, fi + 4 * k * IS);
                    c.i = ld(inp, fi + 4 * k * IS + 4);
                }
                let mut o = [inn[0]; 4];
                for k in 1..4 {
                    let w = C4 { r: dup(tw[ti + k - 1][0]), i: dup(tw[ti + k - 1][1]) };
                    o[k] = cmul(inn[k], w);
                }
                let i0 = cadd(o[0], o[2]);
                let i1 = csub(o[0], o[2]);
                let i2 = cadd(o[1], o[3]);
                let i3 = csub(o[1], o[3]);
                let r0 = cadd(i0, i2);
                let r2 = C4 { r: sub(i0.r, i2.r), i: sub(i2.i, i0.i) };
                let r1 = C4 { r: add(i1.r, i3.i), i: sub(i1.i, i3.r) };
                let r3 = C4 { r: sub(i1.r, i3.i), i: neg(add(i3.r, i1.i)) };
                st(out, fr, r0.r);
                st(out, fr + 4, r0.i);
                st(out, fr + 4 * 2 * ms, r1.r);
                st(out, fr + 4 * (2 * ms + 1), r1.i);
                st(out, fb, r2.r);
                st(out, fb + 4, r2.i);
                st(out, fb - 4 * 2 * ms, r3.r);
                st(out, fb - 4 * (2 * ms - 1), r3.i);
                fi += 8;
                fr += 8;
                fb -= 8;
                ti += 3;
            }
        }
        fin += 4 * 2 * (ms / 2 - 1);
        fo += 4 * 2 * (ms / 2 - 1);
        {
            let q0 = ld(inp, fin);
            let mut q1 = ld(inp, fin + 4 * IS);
            let q2 = ld(inp, fin + 8 * IS);
            let mut q3 = ld(inp, fin + 12 * IS);
            q1 = mul(q1, dup(TW81));
            q3 = mul(q3, dup(TW81));
            let t = sub(q1, q3);
            q3 = add(q1, q3);
            q1 = t;
            st(out, fo, add(q0, q1));
            st(out, fo + 4, neg(add(q2, q3)));
            st(out, fo + 4 * 2 * ms, sub(q0, q1));
            st(out, fo + 4 * (2 * ms + 1), sub(q2, q3));
        }
        fin += 4;
        fo += 4;
        fo += 4 * 3 * ms;
    }
}

/// `dst` holds N/2+1 complex values as interleaved floats (N + 2); `tw` is the flattened su table.
fn last_stage(dst: &mut [f32], src: &[f32], tw: &[f32]) {
    {
        let p = &src[0..4];
        let s0 = p[0] + p[2];
        let s1 = p[1] + p[3];
        dst[0] = s0 + s1;
        dst[1] = s0 - s1;
        dst[2 * (N / 4)] = p[0] - p[2];
        dst[2 * (N / 4) + 1] = p[3] - p[1];
    }
    {
        let p = &src[N - 4..N];
        let s0 = (p[3] - p[1]) * TW81N;
        let s1 = (p[3] + p[1]) * TW81N;
        dst[2 * (N / 8)] = p[0] + s0;
        dst[2 * (N / 8) + 1] = s1 - p[2];
        dst[2 * (N / 8 + N / 4)] = p[0] - s0;
        dst[2 * (N / 8 + N / 4) + 1] = s1 + p[2];
    }
    {
        let mut co = [[0f32; 2]; 4];
        let mut ci = [[0f32; 2]; 4];
        for k in 0..4 {
            co[k] = [src[4 + k], src[8 + k]];
        }
        ci[0] = co[0];
        for k in 1..4 {
            let (twr, twi) = (tw[2 * k], tw[2 * k + 1]);
            let arbr = co[k][0] * twr;
            let aibi = co[k][1] * twi;
            let arbi = co[k][0] * twi;
            let aibr = co[k][1] * twr;
            ci[k] = [arbr - aibi, aibr + arbi];
        }
        let mut t = [[0f32; 2]; 4];
        t[0] = [ci[0][0] + ci[2][0], ci[0][1] + ci[2][1]];
        t[1] = [ci[0][0] - ci[2][0], ci[0][1] - ci[2][1]];
        t[2] = [ci[1][0] + ci[3][0], ci[1][1] + ci[3][1]];
        t[3] = [ci[1][0] - ci[3][0], ci[1][1] - ci[3][1]];
        t[3] = [t[3][1], -t[3][0]];
        let o0 = [t[0][0] + t[2][0], t[0][1] + t[2][1]];
        let o2 = [t[0][0] - t[2][0], -(t[0][1] - t[2][1])];
        let o1 = [t[1][0] + t[3][0], t[1][1] + t[3][1]];
        let o3 = [t[1][0] - t[3][0], -(t[1][1] - t[3][1])];
        for (idx, o) in [(1, o0), (N / 4 + 1, o1), (N / 4 - 1, o3), (N / 2 - 1, o2)] {
            dst[2 * idx] = o[0];
            dst[2 * idx + 1] = o[1];
        }
    }
    {
        let fin = &src[12..];
        let w = &tw[8..];
        let (mut i0, mut i1, mut i2, mut i3) = ([0f32; 4], [0f32; 4], [0f32; 4], [0f32; 4]);
        let (mut o0, mut o1, mut o2, mut o3) = ([0f32; 4], [0f32; 4], [0f32; 4], [0f32; 4]);
        let (mut a0, mut a1, mut b0, mut b1) = ([0f32; 4], [0f32; 4], [0f32; 4], [0f32; 4]);
        for k in 0..4 {
            i0[k] = fin[k];
            i1[k] = fin[4 + k];
            i2[k] = fin[8 + k];
            i3[k] = fin[12 + k];
            a0[k] = w[2 * k];
            a1[k] = w[2 * k + 1];
            b0[k] = w[8 + 2 * k];
            b1[k] = w[8 + 2 * k + 1];
        }
        o0[0] = i0[0];
        o1[0] = i1[0];
        o2[0] = i2[0];
        o3[0] = i3[0];
        for k in 1..4 {
            o0[k] = i0[k] * a0[k] - i1[k] * a1[k];
            o1[k] = i0[k] * a1[k] + i1[k] * a0[k];
            o2[k] = i2[k] * b0[k] - i3[k] * b1[k];
            o3[k] = i2[k] * b1[k] + i3[k] * b0[k];
        }
        i0[0] = o0[0] + o0[2];
        i1[0] = o1[0] + o1[2];
        i0[1] = o0[0] - o0[2];
        i1[1] = o1[0] - o1[2];
        i0[2] = o0[1] + o0[3];
        i1[2] = o1[1] + o1[3];
        i0[3] = o0[1] - o0[3];
        i1[3] = o1[1] - o1[3];
        i2[0] = o2[0] + o2[2];
        i3[0] = o3[0] + o3[2];
        i2[1] = o2[0] - o2[2];
        i3[1] = o3[0] - o3[2];
        i2[2] = o2[1] + o2[3];
        i3[2] = o3[1] + o3[3];
        i2[3] = o2[1] - o2[3];
        i3[3] = o3[1] - o3[3];
        o0[0] = i0[0] + i0[2];
        o0[1] = i1[0] + i1[2];
        o2[2] = i0[0] - i0[2];
        o2[3] = -i1[0] + i1[2];
        o3[2] = i0[1] - i1[3];
        o3[3] = -i1[1] - i0[3];
        o1[0] = i0[1] + i1[3];
        o1[1] = i1[1] - i0[3];
        o0[2] = i2[0] + i2[2];
        o0[3] = i3[0] + i3[2];
        o2[0] = i2[0] - i2[2];
        o2[1] = -i3[0] + i3[2];
        o3[0] = i2[1] - i3[3];
        o3[1] = -i3[1] - i2[3];
        o1[2] = i2[1] + i3[3];
        o1[3] = i3[1] - i2[3];
        dst[4..8].copy_from_slice(&o0);
        dst[4 + N / 2..8 + N / 2].copy_from_slice(&o1);
        dst[N / 2 - 6..N / 2 - 2].copy_from_slice(&o3);
        dst[N / 2 - 6 + N / 2..N / 2 - 2 + N / 2].copy_from_slice(&o2);
    }
    {
        let mut fin = 28usize;
        let mut wi = 24usize;
        let mut fr = 8isize;
        let mut fb = -14isize;
        for _ in 0..((N >> 2) - 8) >> 3 {
            let mut a = [[0f32; 4]; 8];
            for (k, ak) in a.iter_mut().enumerate() {
                *ak = ld(src, fin + 4 * k);
            }
            fin += 32;
            let mut inn = [C4 { r: [0.; 4], i: [0.; 4] }; 4];
            for (k, c) in inn.iter_mut().enumerate() {
                c.r = [a[0][k], a[2][k], a[4][k], a[6][k]];
                c.i = [a[1][k], a[3][k], a[5][k], a[7][k]];
            }
            let mut o = [inn[0]; 4];
            for k in 1..4 {
                let mut t = C4 { r: [0.; 4], i: [0.; 4] };
                for j in 0..4 {
                    t.r[j] = tw[wi + 2 * j];
                    t.i[j] = tw[wi + 2 * j + 1];
                }
                wi += 8;
                o[k] = cmul(inn[k], t);
            }
            let i0 = cadd(o[0], o[2]);
            let i1 = csub(o[0], o[2]);
            let i2 = cadd(o[1], o[3]);
            let i3 = csub(o[1], o[3]);
            let mut r2 = C4 { r: sub(i0.r, i2.r), i: neg(sub(i0.i, i2.i)) };
            let mut r3 = C4 { r: sub(i1.r, i3.i), i: neg(add(i1.i, i3.r)) };
            let r0 = cadd(i0, i2);
            let r1 = C4 { r: add(i1.r, i3.i), i: sub(i1.i, i3.r) };
            r2 = C4 { r: rev4(r2.r), i: rev4(r2.i) };
            r3 = C4 { r: rev4(r3.r), i: rev4(r3.i) };
            let outs = [r0, r1, r3, r2];
            let dsts = [fr, fr + (N / 2) as isize, fb + (N / 2) as isize, fb + N as isize];
            for (o, d) in outs.iter().zip(dsts) {
                for j in 0..4 {
                    dst[d as usize + 2 * j] = o.r[j];
                    dst[d as usize + 2 * j + 1] = o.i[j];
                }
            }
            fr += 8;
            fb -= 8;
        }
    }
}

struct FftTables {
    tw: [[f32; 2]; 120],
    su: [f32; 408],
}

impl FftTables {
    fn new() -> FftTables {
        let mut tw = [[0f32; 2]; 120];
        for (i, t) in tw.iter_mut().enumerate() {
            *t = [NE_TW[2 * i], NE_TW[2 * i + 1]];
        }
        FftTables { tw, su: NE_SU }
    }

    /// In place, Kaldi packing [re0, reN/2, re1, im1, ...].
    fn real_fft(&self, data: &mut [f32; N], a: &mut [f32; N], b: &mut [f32; N + 2]) {
        radix8x4(a, data);
        radix4x4_tw(b, a, 4, 8, &self.tw);
        radix4x4_tw(a, b, 1, 32, &self.tw[24..]);
        last_stage(b, a, &self.su);
        b[N] = b[1];
        b[1] = 0.0;
        b[N + 1] = 0.0;
        b[1] = b[N];
        data.copy_from_slice(&b[..N]);
    }
}

fn mel_scale(f: f32) -> f32 {
    1127.0 * msun_logf(1.0 + f / 700.0)
}

/// Front-end shape: mel bin count and range, taken from the model config.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LfbeSpec {
    pub bins: usize,
    pub low_hz: f32,
    pub high_hz: f32,
}

impl LfbeSpec {
    pub const AED: LfbeSpec = LfbeSpec { bins: 20, low_hz: 100.0, high_hz: 6000.0 };
    pub const WAKE: LfbeSpec = LfbeSpec { bins: 64, low_hz: 80.0, high_hz: 7200.0 };
}

pub struct Lfbe {
    fft: FftTables,
    bins: usize,
    first: Vec<usize>,
    size: Vec<usize>,
    wts: Vec<f32>,
    buf: Vec<i16>,
    w: [f32; N],
    ta: [f32; N],
    tb: [f32; N + 2],
}

impl Default for Lfbe {
    fn default() -> Self {
        Self::new()
    }
}

impl Lfbe {
    pub fn new() -> Lfbe {
        Self::with_spec(LfbeSpec::AED)
    }

    pub fn with_spec(spec: LfbeSpec) -> Lfbe {
        let bins = spec.bins;
        let mut first = vec![0usize; bins];
        let mut size = vec![0usize; bins];
        let mut wts = vec![0f32; bins * 256];
        let bw = 16000.0f32 / N as f32;
        let mlo = mel_scale(spec.low_hz);
        let mhi = mel_scale(spec.high_hz);
        let delta = (mhi - mlo) / (bins + 1) as f32;
        for b in 0..bins {
            let l = mlo + b as f32 * delta;
            let c = mlo + (b + 1) as f32 * delta;
            let r = mlo + (b + 2) as f32 * delta;
            let mut f0: Option<usize> = None;
            let mut li = 0;
            for i in 0..256 {
                let mel = mel_scale(bw * i as f32);
                if mel > l && mel < r {
                    let f = *f0.get_or_insert(i);
                    wts[b * 256 + i - f] = if mel <= c { (mel - l) / (c - l) } else { (r - mel) / (r - c) };
                    li = i;
                }
            }
            let f = f0.unwrap_or(0);
            first[b] = f;
            size[b] = li + 1 - f;
        }
        Lfbe {
            fft: FftTables::new(),
            bins,
            first,
            size,
            wts,
            buf: Vec::with_capacity(FL + HOP),
            w: [0.; N],
            ta: [0.; N],
            tb: [0.; N + 2],
        }
    }

    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Appends `bins()` values per completed frame.
    pub fn push_flat(&mut self, pcm: &[i16], out: &mut Vec<f32>) {
        self.buf.extend_from_slice(pcm);
        let mut pos = 0;
        while self.buf.len() - pos >= FL {
            let at = out.len();
            out.resize(at + self.bins, 0.0);
            self.frame(pos, &mut out[at..]);
            pos += HOP;
        }
        self.buf.drain(..pos);
    }

    /// AED shape only.
    pub fn push(&mut self, pcm: &[i16], out: &mut Vec<[f32; BINS]>) {
        debug_assert_eq!(self.bins, BINS);
        let mut flat = Vec::new();
        self.push_flat(pcm, &mut flat);
        out.extend(flat.chunks_exact(BINS).map(|c| <[f32; BINS]>::try_from(c).unwrap()));
    }

    fn frame(&mut self, pos: usize, out: &mut [f32]) {
        let w = &mut self.w;
        for i in 0..FL {
            w[i] = self.buf[pos + i] as f32;
        }
        let ones = [1f32; FL];
        let mean = -sdot_vfp(&w[..FL], &ones) / FL as f32;
        for v in &mut w[..FL] {
            *v += mean;
        }
        for i in (1..FL).rev() {
            w[i] -= 0.96875 * w[i - 1];
        }
        w[0] -= 0.96875 * w[0];
        for i in 0..FL {
            w[i] *= POVEY[i];
        }
        w[FL..].fill(0.0);
        self.fft.real_fft(w, &mut self.ta, &mut self.tb);
        let first_e = w[0] * w[0];
        let last_e = w[1] * w[1];
        for i in 1..N / 2 {
            let (re, im) = (w[i * 2], w[i * 2 + 1]);
            w[i] = re * re + im * im;
        }
        w[0] = first_e;
        w[N / 2] = last_e;
        for (b, o) in out.iter_mut().enumerate() {
            let (f, n) = (self.first[b], self.size[b]);
            let e = sdot_vfp(&self.wts[b * 256..b * 256 + n], &w[f..f + n]);
            *o = msun_logf(if e < 1.0 { 1.0 } else { e });
        }
    }
}
