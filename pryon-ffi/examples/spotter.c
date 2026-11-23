/* usage: spotter SET_DIR PCM_S16LE_FILE - prints one line per detection. */
#include <stdio.h>
#include <stdlib.h>

#include "pryon.h"

#define OFF(f, v) _Static_assert(offsetof(struct pryon_detection, f) == (v), "pryon_detection." #f " offset")
OFF(keyword, 0);
OFF(kind, 4);
OFF(score, 8);
OFF(begin_sample, 16);
OFF(end_sample, 24);
OFF(afe_known, 32);
OFF(afe_start, 40);
OFF(afe_end, 48);
_Static_assert(sizeof(struct pryon_detection) == 56, "detection size");

static const char *kinds[] = {"ACCEPT", "ACCEPT_AUXILIARY", "NEAR_MISS"};

static void print(pryon_spotter *s, const struct pryon_detection *d) {
    const char *name = pryon_spotter_keyword_name(s, d->keyword);
    const char *kind = d->kind >= 0 && d->kind <= PRYON_DETECTION_NEAR_MISS ? kinds[d->kind] : "?";
    printf("%s %s %.4f begin %llu end %llu", name ? name : "?", kind,
           d->score, (unsigned long long)d->begin_sample, (unsigned long long)d->end_sample);
    if (d->afe_known) printf(" afe %lld %lld", (long long)d->afe_start, (long long)d->afe_end);
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s SET_DIR PCM_S16LE_FILE\n", argv[0]);
        return 2;
    }
    char err[256];
    pryon_spotter *s = pryon_spotter_open(argv[1], 0, err, sizeof err);
    if (!s) {
        fprintf(stderr, "open: %s\n", err);
        return 1;
    }
    FILE *f = fopen(argv[2], "rb");
    if (!f) {
        perror("fopen");
        pryon_spotter_close(s);
        return 1;
    }
    int16_t pcm[800];
    struct pryon_detection d[8];
    size_t got;
    int rc = 0;
    for (;;) {
        got = fread(pcm, sizeof pcm[0], 800, f);
        int n = pryon_spotter_feed(s, pcm, got, d, 8);
        if (n < 0) {
            rc = 1;
            break;
        }
        for (int i = 0; i < n; i++) print(s, &d[i]);
        if (got < 800) break;
    }
    pryon_spotter_reset(s);
    for (int n; (n = pryon_spotter_feed(s, NULL, 0, d, 8)) > 0;)
        for (int i = 0; i < n; i++) print(s, &d[i]);
    fclose(f);
    pryon_spotter_close(s);
    return rc;
}
