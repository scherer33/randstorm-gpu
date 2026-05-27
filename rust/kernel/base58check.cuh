// Base58Check CUDA device functions for Tron (TRC-20) address encoding.
// Input: 21 bytes (0x41 prefix + 20-byte Keccak-256 hash)
// Output: Base58Check encoded string (T... address)
#ifndef BASE58CHECK_CUH
#define BASE58CHECK_CUH

#include <cstdint>

__device__ __constant__ const char BASE58_ALPHABET[58] = {
    '1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','c','d','e','f','g','h','i','j','k','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
};

/// Double SHA-256 for checksum computation (first 4 bytes used as checksum).
/// Optimized for 25-byte input (version + payload).
static __device__ void base58_double_sha256_checksum(
    const uint8_t* data, uint32_t len, uint8_t checksum[4]
) {
    // Inline SHA-256 for checksum — reuse sha256_device if available
    // Since we're in a separate header, we do a minimal double-SHA256
    // This requires sha256.cuh to be included before base58check.cuh
    // We'll use the external sha256_device function

    uint8_t hash1[32];
    sha256_device(data, len, hash1);

    uint8_t hash2[32];
    sha256_device(hash1, 32, hash2);

    checksum[0] = hash2[0];
    checksum[1] = hash2[1];
    checksum[2] = hash2[2];
    checksum[3] = hash2[3];
}

/// Base58 encode raw bytes into output buffer.
/// Returns the number of characters written.
/// output must have space for at least 50 bytes (max for 25-byte input with Base58).
static __device__ int base58_encode(
    const uint8_t* input, uint32_t input_len,
    char* output, uint32_t output_max
) {
    // Count leading zero bytes (they become '1' characters)
    int leading_zeros = 0;
    for (uint32_t i = 0; i < input_len; i++) {
        if (input[i] == 0) leading_zeros++;
        else break;
    }

    // Big number division: convert bytes to base58 digits
    // Work in-place on a copy of the input
    uint8_t temp[25];
    for (uint32_t i = 0; i < input_len; i++) temp[i] = input[i];

    // Count base58 digits
    int digits = 0;
    char reversed[50]; // max output size for 25-byte input

    // Repeatedly divide by 58
    uint32_t start = 0;
    while (start < input_len) {
        // Find first non-zero
        while (start < input_len && temp[start] == 0) start++;
        if (start >= input_len) break;

        // Divide big number by 58
        uint32_t carry = 0;
        for (uint32_t i = start; i < input_len; i++) {
            uint32_t val = carry * 256 + temp[i];
            temp[i] = (uint8_t)(val / 58);
            carry = val % 58;
        }

        reversed[digits++] = BASE58_ALPHABET[carry];
    }

    // Add leading '1's for leading zero bytes
    for (int i = 0; i < leading_zeros && digits < (int)output_max; i++) {
        reversed[digits++] = '1';
    }

    // Reverse into output
    int out_len = 0;
    for (int i = digits - 1; i >= 0 && out_len < (int)output_max; i--) {
        output[out_len++] = reversed[i];
    }

    return out_len;
}

/// Base58Check encode: version + payload + checksum → Base58 string.
/// For Tron: version=0x41, payload=20-byte keccak hash, checksum=first 4 bytes of double-SHA256.
/// Returns number of characters written. Output must have space for at least 50 chars.
static __device__ int base58check_encode(
    uint8_t version, const uint8_t* payload, uint32_t payload_len,
    char* output, uint32_t output_max
) {
    // Build: version + payload + checksum
    uint8_t data[25];
    data[0] = version;
    for (uint32_t i = 0; i < payload_len && i < 20; i++) data[1 + i] = payload[i];

    uint8_t checksum[4];
    base58_double_sha256_checksum(data, 1 + payload_len, checksum);

    // Append checksum
    for (int i = 0; i < 4; i++) data[1 + payload_len + i] = checksum[i];

    return base58_encode(data, 1 + payload_len + 4, output, output_max);
}

#endif // BASE58CHECK_CUH
