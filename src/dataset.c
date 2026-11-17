#include "ig/dataset.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define DOC_REC 13 /* u32 file, u32 off, u32 len, u8 bucket */

static const uint8_t *map_ro(const char *path, size_t *out_len)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ig-ds: cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return NULL;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); /* mapping outlives the fd */
    if (p == MAP_FAILED) {
        fprintf(stderr, "ig-ds: mmap failed for %s\n", path);
        return NULL;
    }
    *out_len = (size_t)st.st_size;
    return p;
}

static uint32_t rd32(const uint8_t *p)
{
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

ig_Dataset *ig_ds_load(const char *index_path)
{
    size_t len;
    const uint8_t *idx = map_ro(index_path, &len);
    if (!idx) return NULL;
    if (len < 12 || memcmp(idx, "IGIX", 4) != 0) {
        fprintf(stderr, "ig-ds: bad index magic in %s\n", index_path);
        munmap((void *)idx, len);
        return NULL;
    }
    uint32_t ver = rd32(idx + 4);
    if (ver != 1) {
        fprintf(stderr, "ig-ds: unsupported index version %u\n", ver);
        munmap((void *)idx, len);
        return NULL;
    }
    ig_Dataset *ds = calloc(1, sizeof *ds);
    if (!ds) return NULL;
    ds->idx = idx;
    ds->idx_len = len;

    size_t pos = 8;
    ds->nfiles = (int)rd32(idx + pos);
    pos += 4;
    ds->files = calloc((size_t)ds->nfiles, sizeof(ig_File));
    if (!ds->files) goto fail;
    for (int i = 0; i < ds->nfiles; i++) {
        uint32_t pl = rd32(idx + pos);
        pos += 4;
        if (pos + pl > len) goto fail;
        char *p = malloc(pl + 1);
        if (!p) goto fail;
        memcpy(p, idx + pos, pl);
        p[pl] = 0;
        pos += pl;
        size_t flen;
        const uint8_t *fm = map_ro(p, &flen);
        if (!fm) { free(p); goto fail; }
        ds->files[i].path = p;
        ds->files[i].map = fm;
        ds->files[i].len = flen;
    }
    ds->docs_off = pos;
    if (len < pos || (len - pos) % DOC_REC != 0) {
        fprintf(stderr, "ig-ds: index length not a multiple of %d\n", DOC_REC);
        goto fail;
    }
    ds->ndocs = (len - pos) / DOC_REC;
    for (size_t i = 0; i < ds->ndocs; i++) {
        const uint8_t *r = idx + pos + i * DOC_REC;
        uint8_t bucket = r[12];
        uint32_t dlen = rd32(r + 8);
        switch (bucket) {
        case 0: ds->n_train++; ds->train_bytes += dlen; break;
        case 1: ds->n_val++; break;
        case 2: ds->n_test++; break;
        default:
            fprintf(stderr, "ig-ds: bad bucket %u at doc %zu\n", bucket, i);
            goto fail;
        }
    }
    size_t cnt[3] = { ds->n_train, ds->n_val, ds->n_test };
    for (int b = 0; b < 3; b++) {
        ds->bucket[b].ids = malloc(sizeof(uint32_t) * (cnt[b] ? cnt[b] : 1));
        ds->bucket[b].cum = malloc(sizeof(uint64_t) * (cnt[b] ? cnt[b] : 1));
        if (!ds->bucket[b].ids || !ds->bucket[b].cum) goto fail;
    }
    if (ds->ndocs > UINT32_MAX) {
        fprintf(stderr, "ig-ds: too many docs for u32 ids\n");
        goto fail;
    }
    for (size_t i = 0; i < ds->ndocs; i++) {
        const uint8_t *r = idx + pos + i * DOC_REC;
        ig_Bucket *bk = &ds->bucket[r[12]];
        bk->bytes += rd32(r + 8);
        bk->ids[bk->n] = (uint32_t)i;
        bk->cum[bk->n] = bk->bytes;
        bk->n++;
    }
    return ds;

fail:
    ig_ds_free(ds);
    return NULL;
}

void ig_ds_free(ig_Dataset *ds)
{
    if (!ds) return;
    if (ds->files) {
        for (int i = 0; i < ds->nfiles; i++) {
            if (ds->files[i].map) munmap((void *)ds->files[i].map, ds->files[i].len);
            free((void *)ds->files[i].path);
        }
        free(ds->files);
    }
    for (int b = 0; b < 3; b++) {
        free(ds->bucket[b].ids);
        free(ds->bucket[b].cum);
    }
    if (ds->idx) munmap((void *)ds->idx, ds->idx_len);
    free(ds);
}

uint64_t ig_splitmix64(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static int doc_at(const ig_Dataset *ds, size_t di, int want_bucket,
                  const uint8_t **out, uint32_t *out_len)
{
    const uint8_t *r = ds->idx + ds->docs_off + di * DOC_REC;
    if (r[12] != (uint8_t)want_bucket) return -1;
    uint32_t fid = rd32(r);
    uint32_t off = rd32(r + 4);
    uint32_t len = rd32(r + 8);
    if (fid >= (uint32_t)ds->nfiles) return -1;
    const ig_File *f = &ds->files[fid];
    if ((size_t)off + len > f->len) return -1; /* corrupt index guard */
    *out = f->map + off;
    *out_len = len;
    return 0;
}

int ig_ds_sample_doc(const ig_Dataset *ds, int bucket, uint64_t *rng,
                     int min_len, const uint8_t **out, uint32_t *out_len)
{
    if (bucket < 0 || bucket > 2) return -1;
    const ig_Bucket *bk = &ds->bucket[bucket];
    if (!bk->n || !bk->bytes) return -1;
    for (int tries = 0; tries < 128; tries++) {
        uint64_t u = ig_splitmix64(rng) % bk->bytes;
        size_t lo = 0, hi = bk->n - 1; /* first i with cum[i] > u */
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (bk->cum[mid] > u) hi = mid;
            else lo = mid + 1;
        }
        if (doc_at(ds, bk->ids[lo], bucket, out, out_len) == 0
            && (int)*out_len >= min_len)
            return 0;
    }
    return -1;
}

int ig_ds_train_doc(const ig_Dataset *ds, uint64_t *rng, int min_len,
                    const uint8_t **out, uint32_t *out_len)
{
    return ig_ds_sample_doc(ds, 0, rng, min_len, out, out_len);
}

int ig_ds_eval_doc(const ig_Dataset *ds, int bucket, size_t i,
                   const uint8_t **out, uint32_t *out_len)
{
    if (bucket < 1 || bucket > 2 || i >= ds->bucket[bucket].n) return -1;
    return doc_at(ds, ds->bucket[bucket].ids[i], bucket, out, out_len);
}
