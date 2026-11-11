#include "ig/data.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint8_t *ig_load_file(const char *path, size_t *out_len)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ig: cannot open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(f, 0, SEEK_END) != 0) { perror("fseek"); exit(1); }
    long len = ftell(f);
    if (len < 0) { perror("ftell"); exit(1); }
    rewind(f);
    uint8_t *buf = malloc((size_t)len + 1);
    if (!buf) { fprintf(stderr, "ig: oom\n"); exit(1); }
    size_t got = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[got] = 0;
    if (out_len) *out_len = got;
    return buf;
}
