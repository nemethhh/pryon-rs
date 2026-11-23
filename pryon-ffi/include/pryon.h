/*
 * pryon.h - C API of pryon-rs: a streaming sound-event (AED) detector (pryon_sound_*) and a
 * wake word spotter (pryon_spotter_*, below).
 *
 * Minimal use:
 *
 *   char err[256];
 *   pryon_sound *s = pryon_sound_open("/path/to/model", NULL, 0, err, sizeof err);
 *   if (!s) { fprintf(stderr, "%s\n", err); return 1; }
 *   struct pryon_sound_result r[4];
 *   int n = pryon_sound_feed(s, pcm, nsamples, r, 4);   // 16 kHz mono s16 PCM
 *   for (int i = 0; i < n; i++)
 *       for (int t = 0; t < r[i].n; t++)
 *           if (r[i].type[t].detected) printf("%s\n", pryon_sound_type_name(s, t));
 *   pryon_sound_close(s);
 *
 * Windows: a window is 159680 samples (998 frames of 10 ms, 9.98 s). Window `index`
 * covers samples [index*159680, (index+1)*159680) counted since open/reset. Its result
 * is returned by the feed call that completes frame 997 of the window, i.e. once 240
 * samples past the window end have been fed.
 *
 * Threads: a handle is used by one thread at a time; separate handles are independent;
 * there is no global state. Model files are read at open only. A feed call can release
 * about 50 network steps at once (when channel-norm statistics form at the window's
 * 100th non-silent frame, or at the frame-997 flush) on top of the steps of its own
 * chunk, so real-time threads should feed from a worker thread. A trailing partial
 * window yields nothing.
 *
 * A panic inside the library (a bug) fails that call and poisons the handle:
 * later feed and set_property calls return -1 and reset does nothing; close it.
 */
#ifndef PRYON_H
#define PRYON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of sound types a handle can serve. */
#define PRYON_SOUND_MAX_TYPES 16

/*
 * Per-type result of one window.
 * samples_above: samples of the window with |x| >= the type's energy threshold
 *   (-32768 never counts).
 * detected: score >= the type's detection threshold AND samples_above >= the type's
 *   minimum sample count.
 * near_miss: not detected AND score >= the type's near-miss threshold (no energy test).
 */
struct pryon_sound_type_result {
    float score;
    uint32_t samples_above;
    uint8_t detected, near_miss;
};

/*
 * One window result. max_energy is the window's maximum energy level over its 998 frames,
 * 0..32767 (per-frame max |x|; a frame containing -32768 contributes nothing). n is the number of valid
 * entries in type[], in the order of pryon_sound_type_name().
 */
struct pryon_sound_result {
    uint64_t index, end_sample;
    int32_t max_energy;
    int32_t n;
    struct pryon_sound_type_result type[PRYON_SOUND_MAX_TYPES];
};

typedef struct pryon_sound pryon_sound;

/*
 * Opens the model directory. types NULL or n 0 selects every type of the model, in
 * AED.json order; otherwise (types non-NULL, n > 0) the n named types (unknown or duplicate name is an error).
 * Negative n with non-NULL types is an error.
 * More than PRYON_SOUND_MAX_TYPES types is an error. Returns NULL on error and, if err
 * is non-NULL and errlen > 0, writes a NUL-terminated message truncated to errlen-1 bytes.
 */
pryon_sound *pryon_sound_open(const char *model_dir, const char *const *types, int n,
                              char *err, size_t errlen);

/*
 * Feeds n samples of 16 kHz mono s16 PCM. Writes up to max completed window results to
 * out and returns how many; results beyond max stay queued and come first on the next
 * call (n 0 drains). Returns -1 on error: s NULL, pcm NULL with n > 0, out NULL with
 * max > 0, or negative max.
 */
int pryon_sound_feed(pryon_sound *s, const int16_t *pcm, size_t n,
                     struct pryon_sound_result *out, int max);

/* Number of types served by the handle; -1 if s is NULL. */
int pryon_sound_type_count(const pryon_sound *s);

/* Name of type i (0 <= i < count), NUL-terminated, valid until close; NULL if out of range. */
const char *pryon_sound_type_name(const pryon_sound *s, int i);

/* Restarts the stream: window counting begins at 0 and queued results are dropped. */
void pryon_sound_reset(pryon_sound *s);

/* Frees the handle; NULL is ignored. */
void pryon_sound_close(pryon_sound *s);

/*
 * ---- Wake word spotter ----
 *
 *   pryon_spotter *w = pryon_spotter_open("/path/to/set", 0, err, sizeof err);
 *   struct pryon_detection d[8];
 *   int n = pryon_spotter_feed(w, pcm, nsamples, d, 8);   // 16 kHz mono s16 PCM
 *   for (int i = 0; i < n; i++)
 *       if (d[i].kind == PRYON_DETECTION_ACCEPT) printf("%s\n", pryon_spotter_keyword_name(w, d[i].keyword));
 *   pryon_spotter_close(w);
 *
 * Same conventions as the sound API: opaque handle, one thread at a time, no global state, files
 * read at open only. Sample positions count samples fed since open; reset does not restart them.
 */

/* Flag for pryon_spotter_open: the stream carries the Echo front end's metadata in its lowest bits. */
#define PRYON_SPOTTER_ECHO_STREAM 1u

enum { PRYON_DETECTION_ACCEPT = 0, PRYON_DETECTION_ACCEPT_AUXILIARY = 1, PRYON_DETECTION_NEAR_MISS = 2 };

/*
 * One detection. keyword indexes pryon_spotter_keyword_name(); kind is a PRYON_DETECTION_* value;
 * begin_sample/end_sample are stream sample positions. afe_known is 1 only with
 * PRYON_SPOTTER_ECHO_STREAM and when the front end's times are known; then afe_start/afe_end are
 * the raw front-end clock values, otherwise all three are 0.
 */
struct pryon_detection {
    int32_t keyword, kind;
    float score;
    uint64_t begin_sample, end_sample;
    int32_t afe_known;
    int64_t afe_start, afe_end;
};

typedef struct pryon_spotter pryon_spotter;

/*
 * Opens the wake set whose pryon.manifest is in model_dir. flags is 0 or PRYON_SPOTTER_ECHO_STREAM;
 * any other bit is an error. A set the engine does not fully understand is refused: the error
 * message names the unsupported element. Returns NULL on error and, if err is non-NULL and
 * errlen > 0, writes a NUL-terminated message truncated to errlen-1 bytes.
 */
pryon_spotter *pryon_spotter_open(const char *model_dir, unsigned flags, char *err, size_t errlen);

/*
 * Feeds n samples of 16 kHz mono s16 PCM. Writes up to max detections to out and returns how
 * many; detections beyond max stay queued and come first on the next call (n 0 drains).
 * Returns -1 on error: s NULL, pcm NULL with n > 0, out NULL with max > 0, or negative max.
 */
int pryon_spotter_feed(pryon_spotter *s, const int16_t *pcm, size_t n, struct pryon_detection *out, int max);

/*
 * Sets a client property (e.g. a kw.cfg threshold override condition); it takes effect from the
 * next 10 ms frame. An unknown or non-UTF-8 name is ignored. Returns 0, or -1 if s or name is NULL.
 */
int pryon_spotter_set_property(pryon_spotter *s, const char *name, int value);

/*
 * Session end: restarts the network and rules; client properties stay and sample counting
 * continues. A pending near miss is queued and returned by the next feed call (n 0 suffices).
 */
void pryon_spotter_reset(pryon_spotter *s);

/* Number of served keywords; -1 if s is NULL. */
int pryon_spotter_keyword_count(const pryon_spotter *s);

/* Name of keyword i (0 <= i < count), NUL-terminated, valid until close; NULL if out of range. */
const char *pryon_spotter_keyword_name(const pryon_spotter *s, int i);

/* The set's checksum string, valid until close; NULL if s is NULL. */
const char *pryon_spotter_checksum(const pryon_spotter *s);

/* Frees the handle; NULL is ignored. */
void pryon_spotter_close(pryon_spotter *s);

/* Library version, a static NUL-terminated string. */
const char *pryon_version(void);

#ifdef __cplusplus
}
#endif

#endif /* PRYON_H */
