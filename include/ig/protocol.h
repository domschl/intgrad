/* Tool-protocol wire format (Decision D2): control sequences inside the raw
 * byte stream. Human-readable view: tools/ig-pretty.
 *
 * INVARIANT (required for unambiguous parsing, enforced in ig-pretty):
 * a sequence ESC b is generated ONLY when b is a non-printable byte
 * (b < 0x20 and not 0x09/0x0A/0x0D). Since role bytes are chosen from that
 * set and ESC only ever appears as part of a generated sequence, no
 * arbitrary payload text can create or corrupt a marker.
 *
 * Grammar (all sequences are ESC followed by one role byte):
 *
 *   stream   := turn*
 *   turn     := SYS  text EOT
 *             | USER text EOT
 *             | ASSISTANT ( text CALL name ARGS json EOT
 *                         | text EOT )*
 *             | TOOL    (RES | ERR) json EOT
 *
 * ESC alone at end of stream = truncated marker (ig-pretty reports it).
 */
#ifndef IG_PROTOCOL_H
#define IG_PROTOCOL_H

#include <stdint.h>

#define IG_ESC       0x00 /* marker prefix */
#define IG_OP_CALL   0x0B /* vertical tab  */
#define IG_OP_ARGS   0x0C /* form feed     */
#define IG_OP_RES    0x0D /* carriage ret. */
#define IG_OP_ERR    0x1C /* file sep      */
#define IG_OP_EOT    0x1D /* group sep     */
#define IG_OP_SYS    0x1E /* record sep    */
#define IG_OP_USER   0x1F /* unit sep      */

/* Role-byte predicate: true iff ESC b is a valid, parseable sequence. */
static inline int ig_is_role(int b)
{
    return b == IG_OP_CALL || b == IG_OP_ARGS || b == IG_OP_RES ||
           b == IG_OP_ERR || b == IG_OP_EOT || b == IG_OP_SYS ||
           b == IG_OP_USER;
}

/* True iff byte b may be embedded literally in marker payload text
 * without making the stream ambiguous (printable + common whitespace). */
static inline int ig_is_payload(int b)
{
    return b >= 0x20 || b == 0x09 || b == 0x0A;
    /* payload is prose/JSON: tab and LF allowed, everything else escaped;
     * raw 0x0D is IG_OP_RES territory and must not appear literally. */
}

#endif /* IG_PROTOCOL_H */
