/*
 * scan_utils.c — C hotspot acceleration for iPhoto scan phase
 *
 * P1: parse_iso8601_to_unix_us   — ISO 8601 datetime parsing
 * P2: compute_file_id_c          — file content hashing (mmap + xxhash)
 * P3: should_include_c           — glob path filtering
 *
 * Build (Linux/macOS):
 *   gcc -O3 -march=native -shared -fPIC -o _scan_utils.so scan_utils.c -lxxhash
 */

#define _GNU_SOURCE   /* timegm() — required on Linux/glibc; not needed on BSD/macOS */

#include <stdint.h>
#include <string.h>
#include <time.h>
#include <limits.h>

/* =====================================================================
 * P1: parse_iso8601_to_unix_us
 * ===================================================================== */

/* Read exactly n decimal digits from s; return -1 on error. */
static int _read_n_digits(const char *s, int n) {
    int v = 0;
    for (int i = 0; i < n; i++) {
        if (s[i] < '0' || s[i] > '9') return -1;
        v = v * 10 + (s[i] - '0');
    }
    return v;
}

/* Parse a sub-second decimal fraction from s, normalised to microseconds.
 * *endp is advanced past all consumed digits. */
static int _parse_subsec_us(const char *s, const char **endp) {
    int digits = 0, us = 0;
    while (digits < 6 && s[digits] >= '0' && s[digits] <= '9') {
        us = us * 10 + (s[digits] - '0');
        digits++;
    }
    /* Pad to 6 decimal places. */
    for (int i = digits; i < 6; i++) us *= 10;
    *endp = s + digits;
    /* Skip any extra digits beyond 6. */
    while (**endp >= '0' && **endp <= '9') (*endp)++;
    return us;
}

/**
 * parse_iso8601_to_unix_us
 *
 * Parse an ISO 8601 string (e.g. "2024-03-15T10:30:00Z" or
 * "2024-03-15T10:30:00+08:00") into a Unix microsecond timestamp.
 * Returns INT64_MIN on parse failure.
 *
 * Supported format: YYYY-MM-DDTHH:MM:SS[.ffffff][Z|+/-HH[:MM]]
 *
 * timegm() is a GNU/BSD extension.  For POSIX-only portability it can be
 * replaced with mktime() adjusted by the local timezone offset.
 */
int64_t parse_iso8601_to_unix_us(const char *s) {
    if (!s || strlen(s) < 19) return INT64_MIN;

    struct tm t = {0};
    t.tm_year = _read_n_digits(s,     4) - 1900;
    t.tm_mon  = _read_n_digits(s + 5, 2) - 1;
    t.tm_mday = _read_n_digits(s + 8, 2);
    t.tm_hour = _read_n_digits(s + 11, 2);
    t.tm_min  = _read_n_digits(s + 14, 2);
    t.tm_sec  = _read_n_digits(s + 17, 2);

    if (t.tm_year < 0 || t.tm_mon < 0 || t.tm_mday < 1 ||
        t.tm_hour < 0 || t.tm_min < 0 || t.tm_sec < 0)
        return INT64_MIN;

    const char *p = s + 19;
    int us = 0;
    int tz_offset_sec = 0;

    /* Optional sub-second fraction. */
    if (*p == '.') { p++; us = _parse_subsec_us(p, &p); }

    /* Timezone designator. */
    if (*p == 'Z' || *p == 'z') {
        tz_offset_sec = 0;
    } else if (*p == '+' || *p == '-') {
        int sign = (*p++ == '+') ? 1 : -1;
        int hh = _read_n_digits(p, 2);
        /* Accept both +HH:MM and +HHMM forms. */
        int mm = 0;
        if (strlen(p) >= 4) {
            const char *mm_start = (p[2] == ':') ? p + 3 : p + 2;
            mm = _read_n_digits(mm_start, 2);
            if (mm < 0) mm = 0;
        }
        if (hh < 0) return INT64_MIN;
        tz_offset_sec = sign * (hh * 3600 + mm * 60);
    }
    /* No timezone marker: treat as UTC (consistent with dateutil.isoparse). */

    time_t epoch = timegm(&t);
    if (epoch == (time_t)-1) return INT64_MIN;

    return (int64_t)(epoch - tz_offset_sec) * 1000000LL + us;
}


/* =====================================================================
 * P2: compute_file_id_c
 * ===================================================================== */

#if !defined(_WIN32)

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <xxhash.h>

#define _THRESHOLD    (2 * 1024 * 1024)   /* 2 MB */
#define _CHUNK_256K   (256 * 1024)         /* 256 KB */

/* Loop pread until count bytes are read or EOF; returns actual bytes read. */
static ssize_t _pread_full(int fd, void *buf, size_t count, off_t offset) {
    ssize_t total = 0;
    while ((size_t)total < count) {
        ssize_t n = pread(fd, (char *)buf + total,
                          count - (size_t)total, offset + total);
        if (n <= 0) break;
        total += n;
    }
    return total;
}

/**
 * compute_file_id_c
 *
 * Compute a 128-bit XXH3 hash of a file (full content for files < 2 MB,
 * sampled head/mid/tail + size for larger files).
 * out_hex must be at least 33 bytes; returns 0 on success, -1 on failure.
 *
 * Output format (32-char hex string) matches Python
 * xxhash.xxh3_128().hexdigest() exactly: high64_hex || low64_hex.
 */
int compute_file_id_c(const char *path, char *out_hex) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return -1; }
    off_t size = st.st_size;

    XXH3_state_t *state = XXH3_createState();
    if (!state) { close(fd); return -1; }
    XXH3_128bits_reset(state);

    int ok = 0;

    if (size > 0 && size <= (off_t)_THRESHOLD) {
        /* Small file: mmap and hash in one shot. */
        void *data = mmap(NULL, (size_t)size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { ok = -1; goto cleanup; }
#ifdef MADV_SEQUENTIAL
        madvise(data, (size_t)size, MADV_SEQUENTIAL);
#endif
        XXH3_128bits_update(state, data, (size_t)size);
        munmap(data, (size_t)size);
    } else if (size > (off_t)_THRESHOLD) {
        /* Large file: sample head + middle + tail, mixing in the file size. */
        uint8_t buf[_CHUNK_256K];
        /* Mix in size as little-endian uint64 (matches Python to_bytes(8, "little")). */
        uint64_t size_le = (uint64_t)size;
        XXH3_128bits_update(state, &size_le, 8);

        ssize_t n;

        /* Head */
        n = _pread_full(fd, buf, _CHUNK_256K, 0);
        if (n < 0) { ok = -1; goto cleanup; }
        XXH3_128bits_update(state, buf, (size_t)n);

        /* Middle */
        off_t mid = (size / 2) - (_CHUNK_256K / 2);
        if (mid < 0) mid = 0;
        n = _pread_full(fd, buf, _CHUNK_256K, mid);
        if (n < 0) { ok = -1; goto cleanup; }
        XXH3_128bits_update(state, buf, (size_t)n);

        /* Tail */
        off_t tail_off = (size > (off_t)_CHUNK_256K)
                         ? (size - (off_t)_CHUNK_256K) : 0;
        n = _pread_full(fd, buf, _CHUNK_256K, tail_off);
        if (n < 0) { ok = -1; goto cleanup; }
        XXH3_128bits_update(state, buf, (size_t)n);
    }
    /* size == 0: hash empty content */

    {
        XXH128_hash_t result = XXH3_128bits_digest(state);
        /* Format matches xxhash.xxh3_128().hexdigest(): high64 || low64 */
        snprintf(out_hex, 33, "%016llx%016llx",
                 (unsigned long long)result.high64,
                 (unsigned long long)result.low64);
    }

cleanup:
    XXH3_freeState(state);
    close(fd);
    return ok;
}

#endif  /* !_WIN32 */


/* =====================================================================
 * P3: should_include_c
 * ===================================================================== */

#if !defined(_WIN32)

#include <fnmatch.h>
#include <stdlib.h>

/*
 * should_include_c
 *
 * Check whether rel_path should be included in the scan:
 *   - not matched by any exclude_globs pattern, AND
 *   - matched by at least one include_globs pattern.
 *
 * include_globs and exclude_globs are NULL-terminated C string arrays.
 * Each entry has already been brace-expanded on the Python side via
 * _expand_cached(). Patterns with a leading "**" prefix are handled by
 * additionally matching without that prefix (since fnmatch(3) does not
 * natively support "**" recursive matching).
 *
 * Returns 1 to include, 0 to exclude.
 */
int should_include_c(
    const char *rel_path,
    const char **include_globs,
    const char **exclude_globs
) {
    /* Exclude check */
    for (int i = 0; exclude_globs[i] != NULL; i++) {
        const char *pat = exclude_globs[i];
        if (fnmatch(pat, rel_path, 0) == 0)
            return 0;
        if (strncmp(pat, "**/", 3) == 0) {
            if (fnmatch(pat + 3, rel_path, 0) == 0)
                return 0;
        }
    }
    /* Include check */
    for (int i = 0; include_globs[i] != NULL; i++) {
        const char *pat = include_globs[i];
        if (fnmatch(pat, rel_path, 0) == 0)
            return 1;
        if (strncmp(pat, "**/", 3) == 0) {
            if (fnmatch(pat + 3, rel_path, 0) == 0)
                return 1;
        }
    }
    return 0;
}

#endif  /* !_WIN32 */
