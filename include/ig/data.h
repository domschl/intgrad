#ifndef IG_DATA_H
#define IG_DATA_H

#include <stddef.h>
#include <stdint.h>

/* Slurp a file into a malloc'd byte buffer. Dies on error. */
uint8_t *ig_load_file(const char *path, size_t *out_len);

#endif /* IG_DATA_H */
