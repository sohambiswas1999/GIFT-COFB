#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <smmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>
#include <emmintrin.h>

#if defined(_MSC_VER)
#define CGAL_ALIGN_16 __declspec(align(16))
#elif defined(__GNUC__)
#define CGAL_ALIGN_16 __attribute__((aligned(16)))
#endif

typedef __m128i V128;

#define SLliepi16(a, o) _mm_slli_epi16(a, o)
#define SRliepi16(a, o) _mm_srli_epi16(a, o)
#define XOR128(a, b) _mm_xor_si128(a, b)
#define AND128(a, b) _mm_and_si128(a, b)
#define OR128(a, b) _mm_or_si128(a, b)
#define STORE128(a, b) _mm_store_si128((V128 *)&(a), b)
#define LOAD128(a) _mm_load_si128((const V128 *)&(a))
#define CONST128(a) _mm_load_si128((const V128 *)&(a))

#define dump(a)          \
    buf[0] = 0;          \
    buf[1] = 0;          \
    buf[2] = 0;          \
    buf[3] = 0;          \
    STORE128(buf[0], a); \
    printf("a1: %08x,a2: %08x,a3: %08x,a4: %08x \n", buf[0], buf[1], buf[2], buf[3])

#define spmv_epi16(in0, in1, out0, out1, idx, m, temp) \
    temp = XOR128(in1, SRliepi16(in0, idx));           \
    temp = AND128(temp, m);                            \
    out1 = XOR128(in1, temp);                          \
    out0 = XOR128(in0, SLliepi16(temp, idx));

#define transpose(a)                          \
    spmv_epi16(a, a, a, a, 3, trmask1, temp); \
    spmv_epi16(a, a, a, a, 6, trmask2, temp); \
    spmv_epi16(a, a, a, a, 9, trmask3, temp);

#define Shuffle(a)                               \
    a = _mm_shuffle_epi8(a, slice_shuffle_mask); \
    spmv_epi16(a, a, a, a, 4, shuffle_mask, temp);

#define Subcell(Slice0, Slice1, Slice2, Slice3)      \
    Slice1 = XOR128(Slice1, AND128(Slice0, Slice2)); \
    Slice0 = XOR128(Slice0, AND128(Slice1, Slice3)); \
    Slice2 = XOR128(Slice2, OR128(Slice0, Slice1));  \
    Slice3 = XOR128(Slice3, Slice2);                 \
    Slice1 = XOR128(Slice1, Slice3);                 \
    Slice2 = XOR128(Slice2, AND128(Slice0, Slice1)); \
    Slice3 = XOR128(Slice3, _mm_set1_epi32(0xffffffff));

#define Permute_bits(Slice0, Slice1, Slice2, Slice3) \
    transpose(Slice0);                               \
    transpose(Slice1);                               \
    transpose(Slice2);                               \
    transpose(Slice3);                               \
    Shuffle(Slice0);                                 \
    Shuffle(Slice1);                                 \
    Shuffle(Slice2);                                 \
    Shuffle(Slice3);                                 \
    Slice0 = Shuffle_S0(Slice0);                     \
    Slice1 = Shuffle_S1(Slice1);                     \
    Slice2 = Shuffle_S2(Slice2);                     \
    Slice3 = Shuffle_S3(Slice3);

#define Add_RK_RC(Slice0, Slice1, Slice2, Slice3, U, V, RC)

#define Shuffle_S0(a) _mm_shuffle_epi8(a, shuffle_mask_S0);
#define Shuffle_S1(a) _mm_shuffle_epi8(a, shuffle_mask_S1);
#define Shuffle_S2(a) _mm_shuffle_epi8(a, shuffle_mask_S2);
#define Shuffle_S3(a) _mm_shuffle_epi8(a, shuffle_mask_S3);

#define DeclareVar                                            \
    V128 S0, S1, S2, S3, temp;                                \
    V128 trmask1 = CONST128(maketrmask1);                     \
    V128 trmask2 = CONST128(maketrmask2);                     \
    V128 trmask3 = CONST128(maketrmask3);                     \
    V128 shuffle_mask = CONST128(make_shuffle_mask);          \
    V128 slice_shuffle_mask = CONST128(makesliceshufflemask); \
    V128 shuffle_mask_S0 = CONST128(make_shuffle_mask_S0);    \
    V128 shuffle_mask_S1 = CONST128(make_shuffle_mask_S1);    \
    V128 shuffle_mask_S2 = CONST128(make_shuffle_mask_S2);    \
    V128 shuffle_mask_S3 = CONST128(make_shuffle_mask_S3);

#define State2Var \
    S0 = LOAD128(input[0]), S1 = LOAD128(input[4]), S2 = LOAD128(input[8]), S0 = LOAD128(input[12]);

CGAL_ALIGN_16 uint32_t input[4] = {0x64786478, 0x64786478, 0x64786478, 0x64786478};

/*Mask for transpose operation and swapmove operation*/
uint16_t maketrmask1[8] = {0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842};       // for first super diagonal
uint16_t maketrmask2[8] = {0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084};       // for 2nd super diagonal
uint16_t maketrmask3[8] = {0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008};       // for 3rd super diagonal
uint16_t make_shuffle_mask[8] = {0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0}; // mask for swapmove used in interlacing shiffle applied to each slice

/*Subsequent epi shuffle masks*/
uint8_t makesliceshufflemask[16] = {0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15};

uint8_t make_shuffle_mask_S0[16] = {0, 3, 2, 1, 4, 7, 6, 5, 8, 11, 10, 9, 12, 15, 14, 13};
uint8_t make_shuffle_mask_S1[16] = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
uint8_t make_shuffle_mask_S2[16] = {2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15};
uint8_t make_shuffle_mask_S3[16] = {3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12};

uint8_t make_shuffle_mask_all_slices[16] = {0, 3, 2, 1, 5, 4, 7, 6, 10, 9, 8, 11, 15, 14, 13, 12};

void main()
{
    CGAL_ALIGN_16 uint32_t buf[4];
    DeclareVar;
    V128 test = LOAD128(input[0]);
    dump(test);
    transpose(test);
    Shuffle(test);
    dump(test);
    test = Shuffle_S3(test);

    dump(test);
}