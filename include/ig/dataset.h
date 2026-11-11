/* Binary dataset index (tools/make_split.py idx.bin) for byte-level training.
 *
 * mmap's the index plus every shard file read-only. Documents carry a split
 * bucket (0=train, 1=val, 2=test) fixed by content hash, so train sampling
 * can never touch held-out docs.
 *
 * idx.bin layout (little-endian, no padding):
 *   'IGIX' u32 ver u32 nfiles { u32 pathlen, path bytes }*
 *   then ndocs records of: u32 file_id, u32 off, u32 len, u8 bucket
 */
#ifndef IG_DATASET_H
#define IG_DATASET_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *path;
    const uint8_t *map;
    size_t len;
} ig_File;

typedef struct {
    ig_File *files;
    int nfiles;
    const uint8_t *idx;   /* mmap'd index; doc records at idx_off */
    size_t idx_len;
    size_t docs_off;
    size_t ndocs;
    size_t n_train, n_val, n_test;
    uint64_t train_bytes;
    size_t *val_ids;  /* doc indices per held-out bucket, for O(1) eval */
    size_t *test_ids;
} ig_Dataset;

ig_Dataset *ig_ds_load(const char *index_path);
void ig_ds_free(ig_Dataset *ds);

/* splitmix64 stream */
uint64_t ig_splitmix64(uint64_t *state);

/* Random train doc with len >= min_len; returns 0 on success. */
int ig_ds_train_doc(const ig_Dataset *ds, uint64_t *rng, int min_len,
                    const uint8_t **out, uint32_t *out_len);

/* Eval doc i (0-based) of bucket (1=val, 2=test); returns 0 on success. */
int ig_ds_eval_doc(const ig_Dataset *ds, int bucket, size_t i,
                   const uint8_t **out, uint32_t *out_len);

#endif /* IG_DATASET_H */
