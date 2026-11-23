//! C API of pryon-rs; see `include/pryon.h`.
#![allow(non_camel_case_types)]

use pryon::{Detection, DetectionKind, Spotter, SpotterModel, SpotterOptions};
use pryon::{SoundDetector, SoundModel, SoundWindow};
use std::collections::VecDeque;
use std::ffi::{c_char, c_int, c_uint, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;

pub const PRYON_SOUND_MAX_TYPES: usize = 16;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct pryon_sound_type_result {
    pub score: f32,
    pub samples_above: u32,
    pub detected: u8,
    pub near_miss: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct pryon_sound_result {
    pub index: u64,
    pub end_sample: u64,
    pub max_energy: i32,
    pub n: i32,
    pub r#type: [pryon_sound_type_result; PRYON_SOUND_MAX_TYPES],
}

pub struct pryon_sound {
    det: SoundDetector,
    names: Vec<CString>,
    pending: VecDeque<SoundWindow>,
    scratch: Vec<SoundWindow>,
    poisoned: bool,
}

/// A handle that a caught panic poisons.
trait Poison {
    fn poisoned(&mut self) -> &mut bool;
}

impl Poison for pryon_sound {
    fn poisoned(&mut self) -> &mut bool {
        &mut self.poisoned
    }
}

/// Runs `f` on the live handle `h` unless it is poisoned; a panic poisons it and yields `fail`.
///
/// # Safety
/// `h` is a live handle used by one thread at a time.
unsafe fn guarded<H: Poison, T>(h: *mut H, fail: T, f: impl FnOnce(&mut H) -> T) -> T {
    let h = unsafe { &mut *h };
    if *h.poisoned() {
        return fail;
    }
    match catch_unwind(AssertUnwindSafe(|| f(h))) {
        Ok(v) => v,
        Err(_) => {
            // the state may be half-updated
            *h.poisoned() = true;
            fail
        }
    }
}

fn write_err(err: *mut c_char, errlen: usize, msg: &str) {
    if err.is_null() || errlen == 0 {
        return;
    }
    let mut n = msg.len().min(errlen - 1);
    while !msg.is_char_boundary(n) {
        n -= 1;
    }
    // SAFETY: the caller guarantees err points to errlen writable bytes; n + 1 <= errlen.
    unsafe {
        std::ptr::copy_nonoverlapping(msg.as_ptr().cast::<c_char>(), err, n);
        *err.add(n) = 0;
    }
}

fn open_impl(model_dir: *const c_char, types: *const *const c_char, n: c_int) -> Result<Box<pryon_sound>, String> {
    if model_dir.is_null() {
        return Err("model_dir is NULL".into());
    }
    // SAFETY: non-NULL, NUL-terminated per the API contract.
    let dir = unsafe { CStr::from_ptr(model_dir) }.to_str().map_err(|_| "model_dir is not UTF-8".to_string())?;
    let mut wanted: Vec<&str> = Vec::new();
    if !types.is_null() && n < 0 {
        return Err("negative type count".into());
    }
    if !types.is_null() && n > 0 {
        for i in 0..n as usize {
            // SAFETY: the caller passes n valid pointers.
            let p = unsafe { *types.add(i) };
            if p.is_null() {
                return Err("type name is NULL".into());
            }
            wanted.push(unsafe { CStr::from_ptr(p) }.to_str().map_err(|_| "type name is not UTF-8".to_string())?);
        }
    }
    let model = Arc::new(SoundModel::load(Path::new(dir)).map_err(|e| e.to_string())?);
    let det = SoundDetector::new(model, &wanted).map_err(|e| e.to_string())?;
    let names = det
        .type_names()
        .iter()
        .map(|s| CString::new(s.as_str()).map_err(|_| format!("type name {s:?} contains a NUL byte")))
        .collect::<Result<Vec<CString>, String>>()?;
    if names.len() > PRYON_SOUND_MAX_TYPES {
        return Err(format!("{} types selected, at most {PRYON_SOUND_MAX_TYPES} supported", names.len()));
    }
    Ok(Box::new(pryon_sound { det, names, pending: VecDeque::new(), scratch: Vec::new(), poisoned: false }))
}

/// # Safety
/// Pointers must satisfy the contract in `pryon.h`.
#[no_mangle]
pub unsafe extern "C" fn pryon_sound_open(
    model_dir: *const c_char,
    types: *const *const c_char,
    n: c_int,
    err: *mut c_char,
    errlen: usize,
) -> *mut pryon_sound {
    match catch_unwind(AssertUnwindSafe(|| open_impl(model_dir, types, n))) {
        Ok(Ok(b)) => Box::into_raw(b),
        Ok(Err(msg)) => {
            write_err(err, errlen, &msg);
            std::ptr::null_mut()
        }
        Err(_) => {
            write_err(err, errlen, "internal error (panic)");
            std::ptr::null_mut()
        }
    }
}

fn convert(w: &SoundWindow) -> pryon_sound_result {
    let mut r = pryon_sound_result {
        index: w.index,
        end_sample: w.end_sample,
        max_energy: w.max_energy,
        n: w.types.len() as i32,
        r#type: [pryon_sound_type_result { score: 0.0, samples_above: 0, detected: 0, near_miss: 0 };
            PRYON_SOUND_MAX_TYPES],
    };
    for (d, t) in r.r#type.iter_mut().zip(&w.types) {
        *d = pryon_sound_type_result {
            score: t.score,
            samples_above: t.samples_above,
            detected: t.detected as u8,
            near_miss: t.near_miss as u8,
        };
    }
    r
}

/// # Safety
/// Pointers must satisfy the contract in `pryon.h`.
#[no_mangle]
pub unsafe extern "C" fn pryon_sound_feed(
    s: *mut pryon_sound,
    pcm: *const i16,
    n: usize,
    out: *mut pryon_sound_result,
    max: c_int,
) -> c_int {
    if s.is_null() || max < 0 || (pcm.is_null() && n > 0) || (out.is_null() && max > 0) {
        return -1;
    }
    // SAFETY: s came from pryon_sound_open and is used by one thread at a time.
    unsafe {
        guarded(s, -1, |s| {
            if n > 0 {
                let pcm = std::slice::from_raw_parts(pcm, n);
                s.scratch.clear();
                s.det.feed(pcm, &mut s.scratch);
                s.pending.extend(s.scratch.drain(..));
            }
            let mut k = 0;
            while k < max as usize {
                let Some(w) = s.pending.pop_front() else { break };
                out.add(k).write(convert(&w));
                k += 1;
            }
            k as c_int
        })
    }
}

/// # Safety
/// `s` must be NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn pryon_sound_type_count(s: *const pryon_sound) -> c_int {
    if s.is_null() {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| unsafe { (*s).names.len() as c_int })).unwrap_or(-1)
}

/// # Safety
/// `s` must be NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn pryon_sound_type_name(s: *const pryon_sound, i: c_int) -> *const c_char {
    if s.is_null() || i < 0 {
        return std::ptr::null();
    }
    catch_unwind(AssertUnwindSafe(|| unsafe {
        #[allow(clippy::needless_borrow)] // explicit: rustc denies an implicit autoref through a raw pointer
        (&(*s).names).get(i as usize).map_or(std::ptr::null(), |c| c.as_ptr())
    }))
    .unwrap_or(std::ptr::null())
}

/// # Safety
/// `s` must be NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn pryon_sound_reset(s: *mut pryon_sound) {
    if s.is_null() {
        return;
    }
    // SAFETY: s is a live handle.
    unsafe {
        guarded(s, (), |s| {
            s.det.reset();
            s.pending.clear();
        })
    }
}

/// # Safety
/// `s` must be NULL or a handle from `pryon_sound_open` not yet closed.
#[no_mangle]
pub unsafe extern "C" fn pryon_sound_close(s: *mut pryon_sound) {
    if !s.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(s) })));
    }
}

pub const PRYON_SPOTTER_ECHO_STREAM: c_uint = 1;
pub const PRYON_DETECTION_ACCEPT: i32 = 0;
pub const PRYON_DETECTION_ACCEPT_AUXILIARY: i32 = 1;
pub const PRYON_DETECTION_NEAR_MISS: i32 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct pryon_detection {
    pub keyword: i32,
    pub kind: i32,
    pub score: f32,
    pub begin_sample: u64,
    pub end_sample: u64,
    pub afe_known: i32,
    pub afe_start: i64,
    pub afe_end: i64,
}

pub struct pryon_spotter {
    sp: Spotter,
    names: Vec<CString>,
    checksum: CString,
    pending: VecDeque<Detection>,
    scratch: Vec<Detection>,
    poisoned: bool,
}

impl Poison for pryon_spotter {
    fn poisoned(&mut self) -> &mut bool {
        &mut self.poisoned
    }
}

fn spotter_open_impl(model_dir: *const c_char, flags: c_uint) -> Result<Box<pryon_spotter>, String> {
    if model_dir.is_null() {
        return Err("model_dir is NULL".into());
    }
    if flags & !PRYON_SPOTTER_ECHO_STREAM != 0 {
        return Err(format!("unknown flag bits {:#x}", flags & !PRYON_SPOTTER_ECHO_STREAM));
    }
    // SAFETY: non-NULL, NUL-terminated per the API contract.
    let dir = unsafe { CStr::from_ptr(model_dir) }.to_str().map_err(|_| "model_dir is not UTF-8".to_string())?;
    let model = Arc::new(SpotterModel::load(Path::new(dir)).map_err(|e| e.to_string())?);
    let opts = SpotterOptions { echo_stream: flags & PRYON_SPOTTER_ECHO_STREAM != 0 };
    let sp = Spotter::new(model.clone(), opts).map_err(|e| e.to_string())?;
    let names = model
        .keywords()
        .iter()
        .map(|s| CString::new(s.as_str()).map_err(|_| format!("keyword {s:?} contains a NUL byte")))
        .collect::<Result<Vec<CString>, String>>()?;
    let checksum = CString::new(model.checksum()).map_err(|_| "checksum contains a NUL byte".to_string())?;
    let (pending, scratch) = (VecDeque::new(), Vec::new());
    Ok(Box::new(pryon_spotter { sp, names, checksum, pending, scratch, poisoned: false }))
}

/// # Safety
/// Pointers must satisfy the contract in `pryon.h`.
#[no_mangle]
pub unsafe extern "C" fn pryon_spotter_open(
    model_dir: *const c_char,
    flags: c_uint,
    err: *mut c_char,
    errlen: usize,
) -> *mut pryon_spotter {
    match catch_unwind(AssertUnwindSafe(|| spotter_open_impl(model_dir, flags))) {
        Ok(Ok(b)) => Box::into_raw(b),
        Ok(Err(msg)) => {
            write_err(err, errlen, &msg);
            std::ptr::null_mut()
        }
        Err(_) => {
            write_err(err, errlen, "internal error (panic)");
            std::ptr::null_mut()
        }
    }
}

fn convert_detection(d: &Detection) -> pryon_detection {
    let (afe_known, afe_start, afe_end) = d.afe.map_or((0, 0, 0), |a| (1, a.start, a.end));
    pryon_detection {
        keyword: d.keyword as i32,
        kind: match d.kind {
            DetectionKind::Accept => PRYON_DETECTION_ACCEPT,
            DetectionKind::AcceptAuxiliary => PRYON_DETECTION_ACCEPT_AUXILIARY,
            DetectionKind::NearMiss => PRYON_DETECTION_NEAR_MISS,
        },
        score: d.score,
        begin_sample: d.begin_sample,
        end_sample: d.end_sample,
        afe_known,
        afe_start,
        afe_end,
    }
}

/// # Safety
/// Pointers must satisfy the contract in `pryon.h`.
#[no_mangle]
pub unsafe extern "C" fn pryon_spotter_feed(
    s: *mut pryon_spotter,
    pcm: *const i16,
    n: usize,
    out: *mut pryon_detection,
    max: c_int,
) -> c_int {
    if s.is_null() || max < 0 || (pcm.is_null() && n > 0) || (out.is_null() && max > 0) {
        return -1;
    }
    // SAFETY: s came from pryon_spotter_open and is used by one thread at a time.
    unsafe {
        guarded(s, -1, |s| {
            if n > 0 {
                let pcm = std::slice::from_raw_parts(pcm, n);
                s.scratch.clear();
                s.sp.feed(pcm, &mut s.scratch);
                s.pending.extend(s.scratch.drain(..));
            }
            let mut k = 0;
            while k < max as usize {
                let Some(d) = s.pending.pop_front() else { break };
                out.add(k).write(convert_detection(&d));
                k += 1;
            }
            k as c_int
        })
    }
}

/// # Safety
/// `s` must be NULL or a live handle; `name` NULL or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn pryon_spotter_set_property(s: *mut pryon_spotter, name: *const c_char, value: c_int) -> c_int {
    if s.is_null() || name.is_null() {
        return -1;
    }
    // SAFETY: s is a live handle; name is NUL-terminated.
    unsafe {
        guarded(s, -1, |s| {
            // Not UTF-8 cannot match any property name: ignored like an unknown one.
            if let Ok(name) = CStr::from_ptr(name).to_str() {
                s.sp.set_property(name, value);
            }
            0
        })
    }
}

/// # Safety
/// `s` must be NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn pryon_spotter_reset(s: *mut pryon_spotter) {
    if s.is_null() {
        return;
    }
    // SAFETY: s is a live handle.
    unsafe {
        guarded(s, (), |s| {
            s.scratch.clear();
            s.sp.reset(&mut s.scratch);
            s.pending.extend(s.scratch.drain(..));
        })
    }
}

/// # Safety
/// `s` must be NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn pryon_spotter_keyword_count(s: *const pryon_spotter) -> c_int {
    if s.is_null() {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| unsafe { (*s).names.len() as c_int })).unwrap_or(-1)
}

/// # Safety
/// `s` must be NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn pryon_spotter_keyword_name(s: *const pryon_spotter, i: c_int) -> *const c_char {
    if s.is_null() || i < 0 {
        return std::ptr::null();
    }
    catch_unwind(AssertUnwindSafe(|| unsafe {
        #[allow(clippy::needless_borrow)] // explicit: rustc denies an implicit autoref through a raw pointer
        (&(*s).names).get(i as usize).map_or(std::ptr::null(), |c| c.as_ptr())
    }))
    .unwrap_or(std::ptr::null())
}

/// # Safety
/// `s` must be NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn pryon_spotter_checksum(s: *const pryon_spotter) -> *const c_char {
    if s.is_null() {
        return std::ptr::null();
    }
    catch_unwind(AssertUnwindSafe(|| unsafe { (*s).checksum.as_ptr() })).unwrap_or(std::ptr::null())
}

/// # Safety
/// `s` must be NULL or a handle from `pryon_spotter_open` not yet closed.
#[no_mangle]
pub unsafe extern "C" fn pryon_spotter_close(s: *mut pryon_spotter) {
    if !s.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(s) })));
    }
}

#[no_mangle]
pub extern "C" fn pryon_version() -> *const c_char {
    catch_unwind(|| concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()).unwrap_or(std::ptr::null())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pryon_testkit::{synth, Clip, TempDir};
    use std::ffi::CStr;
    use std::ptr;

    fn model_dir() -> TempDir {
        let d = TempDir::new("ffi");
        pryon_testkit::write_model_dir(&d, 7, 0.5);
        d
    }

    fn open(dir: &TempDir) -> *mut pryon_sound {
        let path = std::ffi::CString::new(dir.path().to_str().unwrap()).unwrap();
        let mut err = [0 as c_char; 256];
        let s = unsafe { pryon_sound_open(path.as_ptr(), ptr::null(), 0, err.as_mut_ptr(), err.len()) };
        assert!(!s.is_null(), "{:?}", unsafe { CStr::from_ptr(err.as_ptr()) });
        s
    }

    #[test]
    fn open_bad_dir_sets_err() {
        let path = std::ffi::CString::new("/nonexistent/pryon-model").unwrap();
        for len in [256usize, 8] {
            let mut err = vec![0x7f as c_char; len];
            let s = unsafe { pryon_sound_open(path.as_ptr(), ptr::null(), 0, err.as_mut_ptr(), len) };
            assert!(s.is_null());
            let nul = err.iter().position(|&c| c == 0).expect("NUL within errlen");
            assert!(nul > 0);
        }
        let s = unsafe { pryon_sound_open(path.as_ptr(), ptr::null(), 0, ptr::null_mut(), 0) };
        assert!(s.is_null());
    }

    #[test]
    fn open_selects_types_and_names() {
        let d = model_dir();
        let path = std::ffi::CString::new(d.path().to_str().unwrap()).unwrap();
        let a = std::ffi::CString::new("snore").unwrap();
        let b = std::ffi::CString::new("dogBark").unwrap();
        let names = [a.as_ptr(), b.as_ptr()];
        let s = unsafe { pryon_sound_open(path.as_ptr(), names.as_ptr(), 2, ptr::null_mut(), 0) };
        assert!(!s.is_null());
        unsafe {
            assert_eq!(pryon_sound_type_count(s), 2);
            assert_eq!(CStr::from_ptr(pryon_sound_type_name(s, 0)).to_str().unwrap(), "snore");
            assert_eq!(CStr::from_ptr(pryon_sound_type_name(s, 1)).to_str().unwrap(), "dogBark");
            assert!(pryon_sound_type_name(s, 2).is_null());
            pryon_sound_close(s);
        }
        let all = open(&d);
        unsafe {
            assert_eq!(pryon_sound_type_count(all), 12);
            pryon_sound_close(all);
        }
        let bad = std::ffi::CString::new("nope").unwrap();
        let names = [bad.as_ptr()];
        let s = unsafe { pryon_sound_open(path.as_ptr(), names.as_ptr(), 1, ptr::null_mut(), 0) };
        assert!(s.is_null());
    }

    #[test]
    fn feed_queues_beyond_max() {
        let d = model_dir();
        let s = open(&d);
        let mut pcm = Vec::new();
        let noise = synth(Clip::Noise);
        while pcm.len() < 2 * 159_680 + 240 {
            pcm.extend_from_slice(&noise);
        }
        pcm.truncate(2 * 159_680 + 240);
        let mut out: [pryon_sound_result; 1] = unsafe { std::mem::zeroed() };
        unsafe {
            assert_eq!(pryon_sound_feed(s, pcm.as_ptr(), pcm.len(), out.as_mut_ptr(), 1), 1);
            assert_eq!(out[0].index, 0);
            assert_eq!(out[0].n, 12);
            assert_eq!(pryon_sound_feed(s, ptr::null(), 0, out.as_mut_ptr(), 1), 1);
            assert_eq!(out[0].index, 1);
            assert_eq!(out[0].end_sample, 2 * 159_680);
            assert_eq!(pryon_sound_feed(s, ptr::null(), 0, out.as_mut_ptr(), 1), 0);
            pryon_sound_close(s);
        }
    }

    #[test]
    fn null_safe() {
        unsafe {
            pryon_sound_close(ptr::null_mut());
            pryon_sound_reset(ptr::null_mut());
            assert_eq!(pryon_sound_feed(ptr::null_mut(), ptr::null(), 0, ptr::null_mut(), 0), -1);
            assert_eq!(pryon_sound_type_count(ptr::null()), -1);
            assert!(pryon_sound_type_name(ptr::null(), 0).is_null());
            assert!(pryon_sound_open(ptr::null(), ptr::null(), 0, ptr::null_mut(), 0).is_null());
            let d = model_dir();
            let s = open(&d);
            assert_eq!(pryon_sound_feed(s, ptr::null(), 5, ptr::null_mut(), 0), -1);
            assert_eq!(pryon_sound_feed(s, [0i16].as_ptr(), 1, ptr::null_mut(), 1), -1);
            assert_eq!(pryon_sound_feed(s, [0i16].as_ptr(), 1, ptr::null_mut(), -1), -1);
            pryon_sound_close(s);
        }
    }

    #[test]
    fn reset_restarts_window_count() {
        let d = model_dir();
        let s = open(&d);
        let mut pcm = Vec::new();
        let noise = synth(Clip::Noise);
        while pcm.len() < 159_680 + 240 {
            pcm.extend_from_slice(&noise);
        }
        pcm.truncate(159_680 + 240);
        let mut out: [pryon_sound_result; 2] = unsafe { std::mem::zeroed() };
        unsafe {
            assert_eq!(pryon_sound_feed(s, pcm.as_ptr(), pcm.len(), out.as_mut_ptr(), 2), 1);
            pryon_sound_reset(s);
            assert_eq!(pryon_sound_feed(s, pcm.as_ptr(), pcm.len(), out.as_mut_ptr(), 2), 1);
            assert_eq!(out[0].index, 0);
            pryon_sound_close(s);
        }
    }

    #[test]
    fn layout_matches_header() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<pryon_sound_type_result>(), 12);
        assert_eq!(size_of::<pryon_sound_result>(), 216);
        assert_eq!(offset_of!(pryon_sound_type_result, score), 0);
        assert_eq!(offset_of!(pryon_sound_type_result, samples_above), 4);
        assert_eq!(offset_of!(pryon_sound_type_result, detected), 8);
        assert_eq!(offset_of!(pryon_sound_type_result, near_miss), 9);
        assert_eq!(offset_of!(pryon_sound_result, index), 0);
        assert_eq!(offset_of!(pryon_sound_result, end_sample), 8);
        assert_eq!(offset_of!(pryon_sound_result, max_energy), 16);
        assert_eq!(offset_of!(pryon_sound_result, n), 20);
        assert_eq!(offset_of!(pryon_sound_result, r#type), 24);
    }

    fn spotter_set(force: f32) -> TempDir {
        let d = TempDir::new("ffi-spotter");
        pryon_testkit::wakeset::write_davs_set(
            &d,
            &pryon_testkit::wakeset::WakeSetOpts { seed: 1, force_score: Some(force) },
        );
        d
    }

    fn spotter_open(dir: &TempDir, flags: c_uint) -> *mut pryon_spotter {
        let path = std::ffi::CString::new(dir.path().to_str().unwrap()).unwrap();
        let mut err = [0 as c_char; 256];
        let s = unsafe { pryon_spotter_open(path.as_ptr(), flags, err.as_mut_ptr(), err.len()) };
        assert!(!s.is_null(), "{:?}", unsafe { CStr::from_ptr(err.as_ptr()) });
        s
    }

    fn open_err(path: &str, flags: c_uint) -> String {
        let path = std::ffi::CString::new(path).unwrap();
        let mut err = [0x7f as c_char; 512];
        let s = unsafe { pryon_spotter_open(path.as_ptr(), flags, err.as_mut_ptr(), err.len()) };
        assert!(s.is_null());
        unsafe { CStr::from_ptr(err.as_ptr()) }.to_str().unwrap().to_string()
    }

    #[test]
    fn spotter_open_bad_dir_sets_err() {
        let path = std::ffi::CString::new("/nonexistent/pryon-set").unwrap();
        for len in [256usize, 8] {
            let mut err = vec![0x7f as c_char; len];
            let s = unsafe { pryon_spotter_open(path.as_ptr(), 0, err.as_mut_ptr(), len) };
            assert!(s.is_null());
            let nul = err.iter().position(|&c| c == 0).expect("NUL within errlen");
            assert!(nul > 0);
        }
        assert!(unsafe { pryon_spotter_open(path.as_ptr(), 0, ptr::null_mut(), 0) }.is_null());
        // a set the engine refuses names the element
        let d = TempDir::new("ffi-spotter-unsupported");
        pryon_testkit::wakeset::write_davs_set(&d, &pryon_testkit::wakeset::WakeSetOpts { seed: 7, force_score: None });
        let cfg = d.join("BDPGeneratedFiles/kw.cfg.json");
        let text = std::fs::read_to_string(&cfg).unwrap();
        assert!(text.contains(r#""weights": [ 1, 1 ]"#));
        std::fs::write(&cfg, text.replacen(r#""weights": [ 1, 1 ]"#, r#""weights": [ 2, 2 ]"#, 1)).unwrap();
        let msg = open_err(d.path().to_str().unwrap(), 0);
        assert!(msg.contains("unsupported") && msg.contains("HEY_DISNEY"), "{msg}");
        // unknown flag bits are refused, not ignored
        let good = spotter_set(0.99);
        let msg = open_err(good.path().to_str().unwrap(), 0x6);
        assert!(msg.contains("unknown flag bits 0x6"), "{msg}");
    }

    #[test]
    fn spotter_feed_queues_beyond_max() {
        let d = spotter_set(0.99);
        let s = spotter_open(&d, 0);
        let pcm = &synth(Clip::Noise)[..16000 * 6];
        let mut out: [pryon_detection; 1] = unsafe { std::mem::zeroed() };
        unsafe {
            assert_eq!(pryon_spotter_feed(s, pcm.as_ptr(), pcm.len(), out.as_mut_ptr(), 1), 1);
            let first = out[0];
            assert_eq!(first.kind, PRYON_DETECTION_ACCEPT);
            assert_eq!(CStr::from_ptr(pryon_spotter_keyword_name(s, first.keyword)).to_str().unwrap(), "ALEXA");
            assert_eq!(first.afe_known, 0);
            assert_eq!((first.afe_start, first.afe_end), (0, 0));
            let mut rest = Vec::new();
            while pryon_spotter_feed(s, ptr::null(), 0, out.as_mut_ptr(), 1) == 1 {
                rest.push(out[0]);
            }
            assert!(!rest.is_empty(), "expected more than one queued detection");
            assert!(rest[0].end_sample >= first.end_sample);
            // the queued sequence equals what an unbounded feed returns
            let s2 = spotter_open(&d, 0);
            let mut all: Vec<pryon_detection> = vec![std::mem::zeroed(); 64];
            let n = pryon_spotter_feed(s2, pcm.as_ptr(), pcm.len(), all.as_mut_ptr(), 64) as usize;
            assert_eq!(n, 1 + rest.len());
            assert_eq!(all[0].end_sample, first.end_sample);
            for (a, b) in all[1..n].iter().zip(&rest) {
                assert_eq!((a.end_sample, a.score.to_bits(), a.kind), (b.end_sample, b.score.to_bits(), b.kind));
            }
            pryon_spotter_close(s);
            pryon_spotter_close(s2);
        }
    }

    #[test]
    fn spotter_reset_keeps_clock_and_property_is_safe() {
        let d = spotter_set(0.99);
        let s = spotter_open(&d, 0);
        let pcm = &synth(Clip::Noise)[..16000 * 3];
        let mut out: [pryon_detection; 64] = unsafe { std::mem::zeroed() };
        unsafe {
            let a = pryon_spotter_feed(s, pcm.as_ptr(), pcm.len(), out.as_mut_ptr(), 64);
            assert!(a > 0);
            let last_end = out[a as usize - 1].end_sample;
            pryon_spotter_reset(s);
            let b = pryon_spotter_feed(s, pcm.as_ptr(), pcm.len(), out.as_mut_ptr(), 64);
            assert!(b > 0);
            assert!(out[b as usize - 1].end_sample > last_end);
            let name = std::ffi::CString::new("no.such.property").unwrap();
            assert_eq!(pryon_spotter_set_property(s, name.as_ptr(), 1), 0);
            let bad = std::ffi::CString::new(vec![0xffu8, 0xfe]).unwrap();
            assert_eq!(pryon_spotter_set_property(s, bad.as_ptr(), 1), 0);
            pryon_spotter_close(s);
        }
    }

    #[test]
    fn spotter_null_safe() {
        unsafe {
            pryon_spotter_close(ptr::null_mut());
            pryon_spotter_reset(ptr::null_mut());
            assert_eq!(pryon_spotter_feed(ptr::null_mut(), ptr::null(), 0, ptr::null_mut(), 0), -1);
            assert_eq!(pryon_spotter_keyword_count(ptr::null()), -1);
            assert!(pryon_spotter_keyword_name(ptr::null(), 0).is_null());
            assert!(pryon_spotter_checksum(ptr::null()).is_null());
            assert_eq!(pryon_spotter_set_property(ptr::null_mut(), ptr::null(), 0), -1);
            assert!(pryon_spotter_open(ptr::null(), 0, ptr::null_mut(), 0).is_null());
            let d = spotter_set(0.99);
            let s = spotter_open(&d, PRYON_SPOTTER_ECHO_STREAM);
            assert_eq!(pryon_spotter_set_property(s, ptr::null(), 0), -1);
            assert_eq!(pryon_spotter_feed(s, ptr::null(), 5, ptr::null_mut(), 0), -1);
            assert_eq!(pryon_spotter_feed(s, [0i16].as_ptr(), 1, ptr::null_mut(), 1), -1);
            assert_eq!(pryon_spotter_feed(s, [0i16].as_ptr(), 1, ptr::null_mut(), -1), -1);
            assert!(pryon_spotter_keyword_count(s) > 0);
            assert!(pryon_spotter_keyword_name(s, -1).is_null());
            assert!(pryon_spotter_keyword_name(s, pryon_spotter_keyword_count(s)).is_null());
            pryon_spotter_close(s);
        }
    }

    #[test]
    fn spotter_checksum_matches_model() {
        let d = spotter_set(0.99);
        let model = SpotterModel::load(d.path()).unwrap();
        let s = spotter_open(&d, 0);
        unsafe {
            assert_eq!(CStr::from_ptr(pryon_spotter_checksum(s)).to_str().unwrap(), model.checksum());
            assert_eq!(pryon_spotter_keyword_count(s) as usize, model.keywords().len());
            for (i, k) in model.keywords().iter().enumerate() {
                assert_eq!(CStr::from_ptr(pryon_spotter_keyword_name(s, i as c_int)).to_str().unwrap(), k);
            }
            pryon_spotter_close(s);
        }
    }

    #[test]
    fn spotter_echo_stream_afe() {
        let d = spotter_set(0.99);
        let pcm = pryon_testkit::lsb_stream(synth(Clip::Noise)[..16000 * 6].to_vec(), 1000);
        let model = Arc::new(SpotterModel::load(d.path()).unwrap());
        let mut sp = Spotter::new(model, SpotterOptions { echo_stream: true }).unwrap();
        let mut want = Vec::new();
        sp.feed(&pcm, &mut want);
        assert!(want.iter().any(|x| x.afe.is_some()));
        let run = |flags: c_uint| {
            let s = spotter_open(&d, flags);
            let mut out: Vec<pryon_detection> = vec![unsafe { std::mem::zeroed() }; 64];
            let n = unsafe { pryon_spotter_feed(s, pcm.as_ptr(), pcm.len(), out.as_mut_ptr(), 64) };
            unsafe { pryon_spotter_close(s) };
            out.truncate(n as usize);
            out
        };
        let got = run(PRYON_SPOTTER_ECHO_STREAM);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            match w.afe {
                Some(a) => assert_eq!((g.afe_known, g.afe_start, g.afe_end), (1, a.start, a.end)),
                None => assert_eq!((g.afe_known, g.afe_start, g.afe_end), (0, 0, 0)),
            }
        }
        assert!(got.iter().any(|g| g.afe_known == 1));
        let plain = run(0);
        assert!(!plain.is_empty());
        assert!(plain.iter().all(|g| (g.afe_known, g.afe_start, g.afe_end) == (0, 0, 0)));
    }

    #[test]
    fn spotter_layout_matches_header() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<pryon_detection>(), 56);
        assert_eq!(offset_of!(pryon_detection, kind), 4);
        assert_eq!(offset_of!(pryon_detection, score), 8);
        assert_eq!(offset_of!(pryon_detection, begin_sample), 16);
        assert_eq!(offset_of!(pryon_detection, end_sample), 24);
        assert_eq!(offset_of!(pryon_detection, afe_known), 32);
        assert_eq!(offset_of!(pryon_detection, afe_start), 40);
        assert_eq!(offset_of!(pryon_detection, afe_end), 48);
    }

    #[test]
    fn panic_poisons_handle() {
        let d = spotter_set(0.99);
        let s = spotter_open(&d, 0);
        let pcm = [0i16; 160];
        let name = std::ffi::CString::new("x").unwrap();
        unsafe {
            let mut out: [pryon_detection; 1] = std::mem::zeroed();
            assert_eq!(pryon_spotter_feed(s, pcm.as_ptr(), pcm.len(), out.as_mut_ptr(), 1), 0);
            assert_eq!(guarded(s, -1, |_| -> c_int { panic!("injected") }), -1);
            assert_eq!(pryon_spotter_feed(s, pcm.as_ptr(), pcm.len(), out.as_mut_ptr(), 1), -1);
            assert_eq!(pryon_spotter_set_property(s, name.as_ptr(), 1), -1);
            pryon_spotter_reset(s);
            assert!(pryon_spotter_keyword_count(s) > 0);
            pryon_spotter_close(s);
        }
        let m = model_dir();
        let s = open(&m);
        unsafe {
            let mut out: [pryon_sound_result; 1] = std::mem::zeroed();
            assert_eq!(pryon_sound_feed(s, pcm.as_ptr(), pcm.len(), out.as_mut_ptr(), 1), 0);
            guarded(s, (), |_| panic!("injected"));
            assert_eq!(pryon_sound_feed(s, pcm.as_ptr(), pcm.len(), out.as_mut_ptr(), 1), -1);
            pryon_sound_reset(s);
            pryon_sound_close(s);
        }
    }

    #[test]
    fn version_is_crate_version() {
        let v = unsafe { CStr::from_ptr(pryon_version()) };
        assert_eq!(v.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }
}
