# Third-party notices

These notices accompany the Rust reimplementation. They do not grant rights to Amazon firmware,
weights, generated model packages, or recordings. Those assets are not included here.

## Ne10

`pryon/src/lfbe.rs` reimplements the operation order of the real FFT in
[Ne10's NE10_rfft_float32.neonintrinsic.c](https://github.com/projectNe10/Ne10/blob/master/modules/dsp/NE10_rfft_float32.neonintrinsic.c).
The source attribution is Copyright 2014–16 ARM Limited and Contributors.
The project's Copyright 2012–16 ARM Limited and Contributors notice, all three BSD conditions, and the full
disclaimer are reproduced verbatim in [licenses/Ne10-LICENSE.txt](licenses/Ne10-LICENSE.txt). Preserve them with source
and binary distributions, including documentation accompanying any standalone frontend binary.

## Kaldi

The window processing, power spectrum and mel-bank stages in `pryon/src/lfbe.rs` follow Kaldi's feature pipeline;
this file is a new implementation of the observed pipeline and operation order, not an unmodified Kaldi source file.
Kaldi's source carries individual author/copyright notices and Apache License 2.0. Its complete upstream legal
notice and license text are preserved in [licenses/Kaldi-COPYING.txt](licenses/Kaldi-COPYING.txt), obtained from
[Kaldi COPYING](https://github.com/kaldi-asr/kaldi/blob/master/COPYING). That full upstream notice also describes other
Kaldi components; their inclusion in that notice does not mean those components are incorporated here.

## FreeBSD msun / Sun

The ported `msun_logf` in `pryon/src/lfbe.rs` retains this notice:

> Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
> Developed at SunPro, a Sun Microsystems, Inc. business.
> Permission to use, copy, modify, and distribute this software is freely
> granted, provided that this notice is preserved.

## OpenBLAS

The ARMv7 `sdot` accumulation order was observed in the shipped library and independently expressed in `pryon/src/lfbe.rs`.
No OpenBLAS source is included in this code. This is attribution for the algorithm/order studied.

## Julien Pommier neon_mathfun / sse_mathfun

`exp_ps` in `pryon/src/num.rs` reimplements the `exp_ps` polynomial of Julien Pommier's neon_mathfun / sse_mathfun,
Copyright (C) 2011 Julien Pommier, under the zlib licence reproduced in
[licenses/neon_mathfun-zlib.txt](licenses/neon_mathfun-zlib.txt).
