/* Binary dataset index (idx.bin v2, py/intgrad/corpus/make_split.py).
 *
 * mmap's the index plus every shard file read-only. Documents carry a split
 * bucket (0=train, 1=val, 2=test) fixed by content hash, so train sampling
 * can never touch held-out docs.
 *
 * idx.bin v2 (little-endian, no padding; py: intgrad.corpus.make_split):
 *   'IGIX' u32 ver=2
 *   u32 nfiles   { u32 len, path bytes }*  (relative to idx.bin's directory)
 *   u32 nsources { u8 len, name bytes }*   (source id = position)
 *   ndocs records in corpus order: u32 file_id, u32 off, u32 len,
 *                                  u8 bucket, u8 source
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

/* Docs of one split bucket, with an inclusive prefix sum of their byte
 * lengths: sampling u ~ U[0, bytes) and locating it in `cum` picks a doc
 * with probability len/bytes, i.e. every byte of the bucket is equally
 * likely (uniform-per-doc sampling over-weighted short docs ~700x vs
 * books: books are 61% of train bytes but 0.7% of docs). */
typedef struct {
    uint32_t *ids;  /* doc indices in this bucket */
    uint64_t *cum;  /* cum[i] = sum of lens of ids[0..i] */
    size_t n;
    uint64_t bytes;
} ig_Bucket;

#define IG_DS_MAX_SOURCES 32

typedef struct {
    ig_File *files;
    int nfiles;
    const uint8_t *idx;   /* mmap'd index; doc records at idx_off */
    size_t idx_len;
    size_t docs_off;
    size_t ndocs;
    size_t n_train, n_val, n_test;
    uint64_t train_bytes;
    ig_Bucket bucket[3]; /* 0 = train, 1 = val, 2 = test */
    int nsources;
    char source_names[IG_DS_MAX_SOURCES][24];
} ig_Dataset;

ig_Dataset *ig_ds_load(const char *index_path);
void ig_ds_free(ig_Dataset *ds);

/* splitmix64 stream */
uint64_t ig_splitmix64(uint64_t *state);

/* Length-weighted random doc of a bucket (P(doc) = len / bucket bytes)
 * with len >= min_len (shorter picks are redrawn); 0 on success. */
int ig_ds_sample_doc(const ig_Dataset *ds, int bucket, uint64_t *rng,
                     int min_len, const uint8_t **out, uint32_t *out_len);

/* Length-weighted random train doc (= ig_ds_sample_doc, bucket 0). */
int ig_ds_train_doc(const ig_Dataset *ds, uint64_t *rng, int min_len,
                    const uint8_t **out, uint32_t *out_len);

/* Source id (index into source_names) of doc `doc_index` (corpus order). */
int ig_ds_doc_source(const ig_Dataset *ds, size_t doc_index);

/* Eval doc i (0-based) of bucket (1=val, 2=test); returns 0 on success. */
int ig_ds_eval_doc(const ig_Dataset *ds, int bucket, size_t i,
                   const uint8_t **out, uint32_t *out_len);

#endif /* IG_DATASET_H */
