//! C API of pryon-rs; see `include/pryon.h`.
#![allow(non_camel_case_types)]

use pryon::{SoundDetector, SoundModel, SoundWindow};
use std::collections::VecDeque;
use std::ffi::{c_char, c_int, CStr, CString};
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
    Ok(Box::new(pryon_sound { det, names, pending: VecDeque::new(), scratch: Vec::new() }))
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
        r#type: [pryon_sound_type_result { score: 0.0, samples_above: 0, detected: 0, near_miss: 0 }; PRYON_SOUND_MAX_TYPES],
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
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: s came from pryon_sound_open and is used by one thread at a time.
        let s = unsafe { &mut *s };
        if n > 0 {
            let pcm = unsafe { std::slice::from_raw_parts(pcm, n) };
            s.scratch.clear();
            s.det.feed(pcm, &mut s.scratch);
            s.pending.extend(s.scratch.drain(..));
        }
        let mut k = 0;
        while k < max as usize {
            let Some(w) = s.pending.pop_front() else { break };
            unsafe { out.add(k).write(convert(&w)) };
            k += 1;
        }
        k as c_int
    }))
    .unwrap_or(-1)
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
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &mut *s };
        s.det.reset();
        s.pending.clear();
    }));
}

/// # Safety
/// `s` must be NULL or a handle from `pryon_sound_open` not yet closed.
#[no_mangle]
pub unsafe extern "C" fn pryon_sound_close(s: *mut pryon_sound) {
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
    use pryon_testkit::{synth, Clip};
    use std::ffi::CStr;
    use std::ptr;

    fn model_dir() -> tempdir::Dir {
        let d = tempdir::Dir::new();
        pryon_testkit::write_model_dir(d.path(), 7, 0.5);
        d
    }

    mod tempdir {
        use std::path::{Path, PathBuf};
        pub struct Dir(PathBuf);
        impl Dir {
            pub fn new() -> Dir {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                let p = std::env::temp_dir().join(format!("pryon-ffi-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
                std::fs::create_dir_all(&p).unwrap();
                Dir(p)
            }
            pub fn path(&self) -> &Path {
                &self.0
            }
        }
        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    fn open(dir: &tempdir::Dir) -> *mut pryon_sound {
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

    #[test]
    fn version_is_crate_version() {
        let v = unsafe { CStr::from_ptr(pryon_version()) };
        assert_eq!(v.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }
}
