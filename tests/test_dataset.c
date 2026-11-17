/* Dataset sampler regression: documents must be drawn proportional to
 * their length (every byte equally likely), held-out buckets must never
 * leak into train sampling, and min_len must be honored.
 *
 * Builds a synthetic shard + idx.bin (tools/make_split.py layout) in /tmp. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ig/dataset.h"

static void w32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }

int main(void)
{
    char shard[] = "/tmp/ig_test_ds_shard_XXXXXX";
    char index[] = "/tmp/ig_test_ds_idx_XXXXXX";
    int fd1 = mkstemp(shard), fd2 = mkstemp(index);
    if (fd1 < 0 || fd2 < 0) { perror("mkstemp"); return 1; }
    close(fd1);
    close(fd2);

    /* docs: train lens 10, 100, 1000 (filled 'a','b','c'), val 'v' x 500,
     * train 'd' x 3 (shorter than min_len below) */
    const uint32_t lens[5] = { 10, 100, 1000, 500, 3 };
    const uint8_t fill[5] = { 'a', 'b', 'c', 'v', 'd' };
    const uint8_t bucket[5] = { 0, 0, 0, 1, 0 };
    uint32_t offs[5], off = 0;
    FILE *f = fopen(shard, "wb");
    for (int i = 0; i < 5; i++) {
        offs[i] = off;
        for (uint32_t k = 0; k < lens[i]; k++) fputc(fill[i], f);
        off += lens[i];
    }
    fclose(f);
    f = fopen(index, "wb");
    fwrite("IGIX", 1, 4, f);
    w32(f, 1);
    w32(f, 1);
    w32(f, (uint32_t)strlen(shard));
    fwrite(shard, 1, strlen(shard), f);
    for (int i = 0; i < 5; i++) {
        w32(f, 0);
        w32(f, offs[i]);
        w32(f, lens[i]);
        fputc(bucket[i], f);
    }
    fclose(f);

    ig_Dataset *ds = ig_ds_load(index);
    if (!ds) { fprintf(stderr, "FAIL: load\n"); return 1; }
    int rc = 0;
    if (ds->n_train != 4 || ds->n_val != 1 || ds->train_bytes != 1113) {
        fprintf(stderr, "FAIL: counts %zu/%zu bytes %llu\n", ds->n_train,
                ds->n_val, (unsigned long long)ds->train_bytes);
        rc = 1;
    }
    long hits[256] = { 0 };
    uint64_t rng = 99;
    const int N = 200000;
    for (int i = 0; i < N; i++) {
        const uint8_t *d;
        uint32_t len;
        if (ig_ds_train_doc(ds, &rng, 5, &d, &len) != 0) { rc = 1; break; }
        hits[d[0]]++;
    }
    if (hits['v'] || hits['d']) {
        fprintf(stderr, "FAIL: sampled val (%ld) or too-short (%ld) docs\n",
                hits['v'], hits['d']);
        rc = 1;
    }
    /* expected shares among eligible docs: 10/1110, 100/1110, 1000/1110 */
    const char *ks = "abc";
    for (int k = 0; k < 3; k++) {
        double want = lens[k] / 1110.0, got = (double)hits[(int)ks[k]] / N;
        if (fabs(got - want) > 0.01) {
            fprintf(stderr, "FAIL: doc %c share %.4f, want %.4f\n", ks[k], got, want);
            rc = 1;
        }
    }
    const uint8_t *d;
    uint32_t len;
    if (ig_ds_sample_doc(ds, 1, &rng, 1, &d, &len) != 0 || d[0] != 'v') {
        fprintf(stderr, "FAIL: val bucket sampling\n");
        rc = 1;
    }
    ig_ds_free(ds);
    unlink(shard);
    unlink(index);
    if (!rc) puts("dataset: length-weighted sampling ok");
    return rc;
}
