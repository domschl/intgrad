/* ig-pretty: translate intgrad control-byte streams (D2) to human-readable
 * markup and back.  Raw stream stays canonical; this is a view.
 *
 *   ig-pretty           stdin byte stream -> stdout markup (default)
 *   ig-pretty -r        stdin markup      -> stdout byte stream
 *
 * Markup: ESC+role -> \u27eaTAG\u27eb  (CALL/ARGS/RES/ERR/EOT/SYS/USER),
 * any other control byte -> \u27eaX hh\u27eb.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ig/protocol.h"

#define OPEN "\xE2\x9E\xAA"  /* U+27EA ⟪ */
#define CLOSE "\xE2\x9E\xAB" /* U+27EB ⟫ */

typedef struct { uint8_t role; const char *tag; } Tag;
static const Tag tags[] = {
    { IG_OP_CALL, "CALL" }, { IG_OP_ARGS, "ARGS" }, { IG_OP_RES, "RES" },
    { IG_OP_ERR, "ERR" },   { IG_OP_EOT, "EOT" },   { IG_OP_SYS, "SYS" },
    { IG_OP_USER, "USER" },
};
enum { NTAGS = sizeof tags / sizeof *tags };

static unsigned char *read_all(size_t *out_n)
{
    size_t cap = 1 << 16, n = 0;
    unsigned char *buf = malloc(cap);
    if (!buf) { fprintf(stderr, "ig-pretty: oom\n"); exit(1); }
    size_t got;
    while ((got = fread(buf + n, 1, cap - n, stdin)) > 0) {
        n += got;
        if (n == cap) {
            cap *= 2;
            buf = realloc(buf, cap);
            if (!buf) { fprintf(stderr, "ig-pretty: oom\n"); exit(1); }
        }
    }
    *out_n = n;
    return buf;
}

static int is_open(const unsigned char *b, size_t i, size_t n)
{
    return i + 2 < n && b[i] == 0xE2 && b[i + 1] == 0x9E && b[i + 2] == 0xAA;
}

static void pretty(const unsigned char *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (b[i] == IG_ESC) {
            if (i + 1 >= n) { fputs(OPEN "TRUNC" CLOSE, stdout); break; }
            if (ig_is_role(b[i + 1])) {
                int k;
                for (k = 0; k < NTAGS; k++)
                    if (tags[k].role == b[i + 1]) break;
                if (k < NTAGS) {
                    printf(OPEN "%s" CLOSE, tags[k].tag);
                    i++;
                    continue;
                }
            }
            printf(OPEN "X00" CLOSE); /* stray ESC: protocol violation;   */
            continue;                 /* do NOT consume the next byte     */
        }
        if (b[i] < 0x20 && b[i] != 0x09 && b[i] != 0x0A && b[i] != 0x0D) {
            printf(OPEN "X%02X" CLOSE, b[i]); /* bare control byte */
        } else {
            fputc(b[i], stdout);
        }
    }
    /* no trailing newline: pretty ⇄ raw is an exact byte round-trip.
     * For terminal display, pipe through: `ig-pretty < f | ...; echo` */
}

/* At position i there is an opening bracket. Try to match a full tag.
 * Returns bytes consumed (including brackets) or 0.
 * kind: 1 = symbolic marker (emit ESC + *byte_out),
 *       0 = raw byte escape (emit *byte_out literally). */
static int match_tag(const unsigned char *b, size_t i, size_t n,
                     uint8_t *byte_out, int *kind)
{
    const size_t o = 3, c = 3;
    size_t p = i + o;
    for (int k = 0; k < NTAGS; k++) {
        size_t tl = strlen(tags[k].tag);
        if (p + tl + c <= n && memcmp(b + p, tags[k].tag, tl) == 0 &&
            memcmp(b + p + tl, CLOSE, c) == 0) {
            *byte_out = tags[k].role;
            *kind = 1;
            return (int)(o + tl + c);
        }
    }
    /* X<hex2> form: OPEN 'X' HH CLOSE -> one literal byte */
    if (p + 3 + c <= n && b[p] == 'X' &&
        isxdigit(b[p + 1]) && isxdigit(b[p + 2]) &&
        memcmp(b + p + 3, CLOSE, c) == 0) {
        char hx[3] = { (char)b[p + 1], (char)b[p + 2], 0 };
        *byte_out = (uint8_t)strtol(hx, NULL, 16);
        *kind = 0;
        return (int)(o + 3 + c);
    }
    return 0;
}

static void raw(const unsigned char *b, size_t n)
{
    for (size_t i = 0; i < n; ) {
        uint8_t byte_val;
        int kind;
        if (is_open(b, i, n)) {
            int used = match_tag(b, i, n, &byte_val, &kind);
            if (used) {
                if (kind) fputc(IG_ESC, stdout);
                fputc(byte_val, stdout);
                i += (size_t)used;
                continue;
            }
        }
        fputc(b[i++], stdout);
    }
}

int main(int argc, char **argv)
{
    int reverse = argc > 1 && strcmp(argv[1], "-r") == 0;
    size_t n;
    unsigned char *buf = read_all(&n);
    if (reverse) raw(buf, n);
    else pretty(buf, n);
    free(buf);
    return 0;
}
