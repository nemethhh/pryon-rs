//! The Echo front end's clock in the stream's lowest bits: libpryon's AFE metadata decoder
//! (audio_metadata_decoder.cpp node 0x999380, decoder 0x103f080, lookup 0x1040458, result times 0x6d2bb0).

use super::AfeTimes;

const SYNC: u128 = 0xa5;
const FRAME: u64 = 128;

/// One decoded frame that carried a clock: its first sample (stream count) and the 16-bit clock.
#[derive(Clone, Copy, Default)]
struct Rec {
    start: u64,
    clock: u16,
}

pub(crate) struct AfeClock {
    /// The last 128 low bits, newest in bit 0.
    win: u128,
    /// Samples into the window since its reset; a frame is looked for from the 128th on.
    count: u32,
    /// Version byte of the last accepted frame; 0xff before the first.
    version: u8,
    /// Samples pushed; never restarts (libpryon keys frames by the stream's sample index).
    fed: u64,
    /// `fed` when the first frame of any version decoded since the last flush.
    first: Option<u64>,
    ring: Box<[Rec]>,
    head: usize,
    len: usize,
}

/// Payload bits after sync and version (the `magic` field) each major-2/3 minor version must carry.
fn payload_bits(major: u8, minor: u8) -> u32 {
    match (major, minor) {
        (2, 0) => 73,
        (2, 1) => 86,
        (2, 2) => 103,
        (3, 0) => 59,
        _ => 104,
    }
}

impl AfeClock {
    /// `records`: frames kept for lookups (libpryon keeps every keyword start it can report).
    pub fn new(records: usize) -> AfeClock {
        let ring = vec![Rec::default(); records.max(1)].into_boxed_slice();
        AfeClock { win: 0, count: 0, version: 0xff, fed: 0, first: None, ring, head: 0, len: 0 }
    }

    pub fn push(&mut self, pcm: &[i16]) {
        for &x in pcm {
            self.win = self.win << 1 | (x & 1) as u128;
            self.fed += 1;
            if self.count < FRAME as u32 - 1 {
                self.count += 1;
            } else if self.win >> 120 == SYNC {
                self.frame();
            }
        }
    }

    /// Arrival bits `p..p + n` of the window, the first one most significant.
    fn field(&self, p: u32, n: u32) -> u32 {
        (self.win >> (128 - p - n)) as u32 & ((1u32 << n) - 1)
    }

    /// A sync byte at the window's start (0x103f080): the frame counts if its version is one libpryon parses, its
    /// layout's checksum (the number of 1 bits before it) holds and the version did not just change.
    fn frame(&mut self) {
        let v = self.field(8, 8) as u8;
        let (major, minor) = (v >> 5, v & 31);
        let (end, clock) = match major {
            0 | 1 if minor != 0 => return,
            0 => (48, None),
            1 => (25, None),
            2 | 3 => {
                let m = self.field(16, 7);
                if m != payload_bits(major, minor) {
                    return;
                }
                (16 + m, Some(self.field(if major == 3 { 29 } else { 72 }, 16) as u16))
            }
            _ => return,
        };
        if self.field(end, 8) != (self.win >> (128 - end)).count_ones() {
            return;
        }
        let first = self.version == 0xff;
        let changed = !first && self.version != v;
        self.version = v;
        // the first frame after a version change is logged and dropped
        if changed {
            return;
        }
        self.first.get_or_insert(self.fed);
        if let Some(clock) = clock {
            let cap = self.ring.len();
            self.ring[self.head] = Rec { start: self.fed - FRAME, clock };
            self.head = (self.head + 1) % cap;
            self.len = (self.len + 1).min(cap);
        }
        (self.win, self.count) = (0, 0);
    }

    /// Session end or a new decoder: everything but the sample count restarts (0x999a7c, 0x103f00e).
    pub fn flush(&mut self) {
        (self.win, self.count, self.version, self.first, self.head, self.len) = (0, 0, 0xff, None, 0, 0);
    }

    /// Times of a result built when `at` samples had reached the decoder, for keyword start and end samples
    /// (kwSampleStartIndex, kwSampleEndIndex); `None` when no frame had decoded (has_audio_lsb_metadata 0).
    pub fn times(&self, start: u64, end: u64, at: u64) -> Option<AfeTimes> {
        if self.first? > at {
            return None;
        }
        Some(AfeTimes { start: self.before(start, at).into(), end: self.before(end, at).into() })
    }

    /// Clock of the newest frame decoded by `at` that starts at or before sample `q`; 0 without one or for
    /// sample 0 (findMetadataRecordBeforeSampleIndex's zeroed record).
    fn before(&self, q: u64, at: u64) -> u16 {
        if q == 0 {
            return 0;
        }
        let cap = self.ring.len();
        (1..=self.len)
            .map(|k| self.ring[(self.head + cap - k) % cap])
            .find(|r| r.start + FRAME <= at && r.start <= q)
            .map_or(0, |r| r.clock)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pryon_testkit::{lsb_stream, lsb_stream_with, synth, Clip, LsbFrame};

    fn clock(pcm: &[i16], chunk: usize) -> AfeClock {
        let mut c = AfeClock::new(4096);
        for p in pcm.chunks(chunk) {
            c.push(p);
        }
        c
    }

    fn noise(n: usize) -> Vec<i16> {
        synth(Clip::Noise)[..n].to_vec()
    }

    #[test]
    fn afe_decodes_synthetic_clock() {
        let pcm = lsb_stream(noise(64000), 1234);
        for chunk in [1, 7, 128, 160, 3333, pcm.len()] {
            let c = clock(&pcm, chunk);
            // frame k starts at 128 k and carries 1234 + 8 k
            for (q, want) in [(1, 1234), (127, 1234), (128, 1242), (12345, 1234 + 8 * 96), (63999, 1234 + 8 * 499)] {
                assert_eq!(c.before(q, pcm.len() as u64), want, "chunk {chunk} sample {q}");
            }
            assert_eq!(c.times(1000, 5000, 64000), Some(AfeTimes { start: 1234 + 56, end: 1234 + 8 * 39 }));
        }
    }

    #[test]
    fn afe_frames_decoded_by_the_result() {
        let c = clock(&lsb_stream(noise(16000), 0), 160);
        // the frame ending at sample 960 is the newest a result at 960 sees, even for later samples
        assert_eq!(c.times(160, 5000, 960), Some(AfeTimes { start: 8, end: 8 * 6 }));
        assert_eq!(c.times(160, 5000, 1023), Some(AfeTimes { start: 8, end: 8 * 6 }));
        assert_eq!(c.times(160, 5000, 1024), Some(AfeTimes { start: 8, end: 8 * 7 }));
        assert_eq!(c.times(160, 5000, 127), None, "no frame decoded yet");
        assert_eq!(c.times(0, 300, 1024).map(|t| t.start), Some(0), "sample 0 reads a zeroed record");
    }

    #[test]
    fn afe_wraps_16_bit() {
        let c = clock(&lsb_stream(noise(16000), 65530), 1600);
        assert_eq!(c.before(10, 16000), 65530);
        assert_eq!(c.before(128, 16000), 2, "65538 mod 65536");
        assert_eq!(c.before(15999, 16000), ((65530 + 8 * 124) % 65536) as u16);
    }

    #[test]
    fn afe_none_without_marks() {
        for clip in Clip::ALL {
            let pcm = synth(clip);
            let c = clock(&pcm, 1600);
            // libpryon too finds a frame without a clock in this one's low bits (synth-afe/*quiet_tones)
            let want = (clip == Clip::QuietTones).then_some(AfeTimes { start: 0, end: 0 });
            assert_eq!(c.times(16000, 32000, pcm.len() as u64), want, "{}", clip.name());
        }
    }

    #[test]
    fn afe_corrupt_frames_skipped() {
        let pcm = lsb_stream_with(noise(16000), 100, |k| if k % 2 == 1 { LsbFrame::Corrupt } else { LsbFrame::V3_3 });
        let c = clock(&pcm, 1600);
        // odd frames fail their checksum: the even frame before stands in
        assert_eq!(c.before(128 * 5 + 3, 16000), 100 + 8 * 4);
        assert_eq!(c.before(128 * 6, 16000), 100 + 8 * 6);
    }

    #[test]
    fn afe_versions() {
        // 2.x carries the clock at bits 72..88, 0.0 and 1.0 none; each version change drops its first frame
        let kinds = |k: usize| match k / 10 {
            0 => LsbFrame::V3_3,
            1 => LsbFrame::V2_0,
            2 => LsbFrame::V2_4,
            3 => LsbFrame::V3_0,
            4 => LsbFrame::V0_0,
            5 => LsbFrame::V1_0,
            _ => LsbFrame::Clean,
        };
        let c = clock(&lsb_stream_with(noise(128 * 70), 0, kinds), 500);
        let at = 128 * 70;
        for (frame, want) in [(9, 9), (10, 9), (11, 11), (20, 19), (21, 21), (30, 29), (31, 31), (39, 39), (55, 39)] {
            assert_eq!(c.before(128 * frame, at), 8 * want as u16, "frame {frame}");
        }
        let only_v0 = clock(&lsb_stream_with(noise(4096), 0, |_| LsbFrame::V0_0), 500);
        assert_eq!(only_v0.times(1000, 2000, 4096), Some(AfeTimes { start: 0, end: 0 }), "decoded, no clock");
    }

    #[test]
    fn afe_alternating_versions_keep_only_the_first_frame() {
        let pcm = lsb_stream_with(noise(16000), 7, |k| if k % 2 == 0 { LsbFrame::V3_3 } else { LsbFrame::V2_4 });
        let c = clock(&pcm, 1600);
        assert_eq!(c.before(15000, 16000), 7);
        assert_eq!(c.len, 1);
    }

    #[test]
    fn afe_frames_found_off_grid() {
        let mut pcm = noise(16000 + 37);
        let marked = lsb_stream(pcm[37..].to_vec(), 500);
        pcm[37..].copy_from_slice(&marked);
        let c = clock(&pcm, 1600);
        assert_eq!(c.before(37 + 128 * 3, 16037), 500 + 24);
        assert_eq!(c.before(37 + 128 * 3 - 1, 16037), 500 + 16);
    }

    #[test]
    fn afe_flush_restarts_decoder() {
        let pcm = lsb_stream(noise(32000), 0);
        let mut c = AfeClock::new(4096);
        c.push(&pcm[..10000]);
        c.flush();
        assert_eq!(c.times(5000, 9000, 10000), None);
        c.push(&pcm[10000..10100]);
        // the frame under way at the flush is lost; the next one (10112..10240) is the first again
        c.push(&pcm[10100..]);
        assert_eq!(c.times(5000, 10200, 10239), None);
        assert_eq!(c.times(5000, 10200, 10240), Some(AfeTimes { start: 0, end: 8 * 79 }));
        assert_eq!(c.before(20000, 32000), 8 * 156);
    }

    #[test]
    fn afe_ring_keeps_newest() {
        let c = {
            let mut c = AfeClock::new(10);
            c.push(&lsb_stream(noise(128 * 30), 0));
            c
        };
        assert_eq!(c.before(128 * 25, 128 * 30), 8 * 25);
        assert_eq!(c.before(128 * 20, 128 * 30), 8 * 20);
        assert_eq!(c.before(128 * 19, 128 * 30), 0, "older than the ring");
    }

    #[test]
    fn afe_any_pcm() {
        // every 16-bit value, then long runs of each LSB pattern: no panic
        let mut c = AfeClock::new(8);
        let all: Vec<i16> = (i16::MIN..=i16::MAX).collect();
        c.push(&all);
        c.push(&[-1; 4096]);
        c.push(&[i16::MIN; 4096]);
        let _ = c.times(u64::MAX, u64::MAX, u64::MAX);
        let _ = c.times(0, 0, 0);
    }
}
