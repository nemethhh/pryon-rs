/* usage: sound MODEL_DIR PCM_S16LE_FILE - prints one line per completed window. */
#include <stdio.h>
#include <stdlib.h>

#include "pryon.h"

#define OFF(t, f, v) _Static_assert(offsetof(struct t, f) == (v), #t "." #f " offset")
OFF(pryon_sound_type_result, score, 0);
OFF(pryon_sound_type_result, samples_above, 4);
OFF(pryon_sound_type_result, detected, 8);
OFF(pryon_sound_type_result, near_miss, 9);
OFF(pryon_sound_result, index, 0);
OFF(pryon_sound_result, end_sample, 8);
OFF(pryon_sound_result, max_energy, 16);
OFF(pryon_sound_result, n, 20);
OFF(pryon_sound_result, type, 24);
_Static_assert(sizeof(struct pryon_sound_type_result) == 12, "type_result size");
_Static_assert(sizeof(struct pryon_sound_result) == 216, "result size");

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_DIR PCM_S16LE_FILE\n", argv[0]);
        return 2;
    }
    char err[256];
    pryon_sound *s = pryon_sound_open(argv[1], NULL, 0, err, sizeof err);
    if (!s) {
        fprintf(stderr, "open: %s\n", err);
        return 1;
    }
    FILE *f = fopen(argv[2], "rb");
    if (!f) {
        perror("fopen");
        pryon_sound_close(s);
        return 1;
    }
    int16_t pcm[800];
    struct pryon_sound_result r[4];
    size_t got;
    int rc = 0;
    for (;;) {
        got = fread(pcm, sizeof pcm[0], 800, f);
        int n = pryon_sound_feed(s, pcm, got, r, 4);
        if (n < 0) {
            rc = 1;
            break;
        }
        for (int i = 0; i < n; i++) {
            printf("window %llu end %llu max_energy %d", (unsigned long long)r[i].index,
                   (unsigned long long)r[i].end_sample, (int)r[i].max_energy);
            for (int t = 0; t < r[i].n; t++)
                printf(" %s=%.4f/%u%s", pryon_sound_type_name(s, t), r[i].type[t].score,
                       (unsigned)r[i].type[t].samples_above,
                       r[i].type[t].detected ? "!" : (r[i].type[t].near_miss ? "?" : ""));
            printf("\n");
        }
        if (got < 800) break;
    }
    fclose(f);
    pryon_sound_close(s);
    return rc;
}
