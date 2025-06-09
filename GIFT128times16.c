#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <smmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>
#include <emmintrin.h>
#include "measurement.h"

#if defined(_MSC_VER)
#define CGAL_ALIGN_16 __declspec(align(64))
#elif defined(__GNUC__)
#define CGAL_ALIGN_16 __attribute__((aligned(64)))
#endif

typedef __m512i V256;

#define SLliepi16(a, o) _mm512_slli_epi16(a, o)
#define SRliepi16(a, o) _mm512_srli_epi16(a, o)
#define XOR128(a, b) _mm512_xor_si512(a, b)
#define AND128(a, b) _mm512_and_si512(a, b)
#define OR128(a, b) _mm512_or_si512(a, b)
#define STORE128(a, b) _mm512_store_si512((V256 *)&(a), b)
#define LOAD128(a) _mm512_load_si512((const V256 *)&(a))
#define CONST128(a) _mm512_load_si512((const V256 *)&(a))
#define ROL16in128(a, o) _mm512_or_si512(_mm512_slli_epi16(a, 16 - (o)), _mm512_srli_epi16(a, o))

#define dump(a)          \
    buf[0] = 0;          \
    buf[1] = 0;          \
    buf[2] = 0;          \
    buf[3] = 0;          \
    buf[4] = 0;          \
    buf[5] = 0;          \
    buf[6] = 0;          \
    buf[7] = 0;          \
    buf[0+8] = 0;          \
    buf[1+8] = 0;          \
    buf[2+8] = 0;          \
    buf[3+8] = 0;          \
    buf[4+8] = 0;          \
    buf[5+8] = 0;          \
    buf[6+8] = 0;          \
    buf[7+8] = 0;		 \
    STORE128(buf[0], a); \
    printf("a1: %08x,a2: %08x,a3: %08x,a4: %08x,a5: %08x,a6: %08x,a7: %08x,a8: %08x \n", buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);\
    printf("a8: %08x,a9: %08x,a10: %08x,a11: %08x,a12: %08x,a13: %08x,a14: %08x,a15: %08x \n", buf[0+8], buf[1+8], buf[2+8], buf[3+8], buf[4+8], buf[5+8], buf[6+8], buf[7+8])

#define dump_key() \
    dump(K[0]);    \
    dump(K[1]);    \
    dump(K[2]);    \
    dump(K[3]);
#define dump_state() \
    dump(S0);        \
    dump(S1);        \
    dump(S2);        \
    dump(S3);

#define spmv_epi16(in0, in1, out0, out1, idx, m, temp) \
    temp = XOR128(in1, SRliepi16(in0, idx));           \
    temp = AND128(temp, m);                            \
    out1 = XOR128(in1, temp);                          \
    out0 = XOR128(in0, SLliepi16(temp, idx));

#define transpose(a)                          \
    spmv_epi16(a, a, a, a, 3, trmask1, temp); \
    spmv_epi16(a, a, a, a, 6, trmask2, temp); \
    spmv_epi16(a, a, a, a, 9, trmask3, temp);

#define Shuffle(a)                                  \
    a = _mm512_shuffle_epi8(a, slice_shuffle_mask); \
    spmv_epi16(a, a, a, a, 4, shuffle_mask, temp);

#define Subcell(Slice0, Slice1, Slice2, Slice3)      \
    Slice1 = XOR128(Slice1, AND128(Slice0, Slice2)); \
    Slice0 = XOR128(Slice0, AND128(Slice1, Slice3)); \
    Slice2 = XOR128(Slice2, OR128(Slice0, Slice1));  \
    Slice3 = XOR128(Slice3, Slice2);                 \
    Slice1 = XOR128(Slice1, Slice3);                 \
    Slice2 = XOR128(Slice2, AND128(Slice0, Slice1)); \
    Slice3 = XOR128(Slice3, _mm512_set1_epi32(0xffffffff));

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

#define Add_RK_RC(Slice0, Slice1, Slice2, Slice3, U, V, RC) \
    Slice2 = XOR128(Slice2, U);                             \
    Slice1 = XOR128(Slice1, V);                             \
    Slice3 = XOR128(Slice3, _mm512_set_epi8(0x80, 0x00, 0x00, RC, 0x80, 0x00, 0x00, RC, 0x80, 0x00, 0x00, RC, 0x80, 0x00, 0x00, RC, 0x80, 0x00, 0x00, RC, 0x80, 0x00, 0x00, RC, 0x80, 0x00, 0x00, RC, 0x80, 0x00, 0x00, RC,0x80, 0x00, 0x00, RC, 0x80, 0x00, 0x00, RC, 0x80, 0x00, 0x00, RC, 0x80, 0x00, 0x00, RC, 0x80, 0x00, 0x00, RC, 0x80, 0x00, 0x00, RC, 0x80, 0x00, 0x00, RC, 0x80, 0x00, 0x00, RC));

#define Key_Update(Round_number)                              \
    temp_1 = ROL16in128(K[(3 + (3 * Round_number)) % 4], 2);  \
    temp_2 = ROL16in128(K[(3 + (3 * Round_number)) % 4], 12); \
    K[(3 + (3 * Round_number)) % 4] = OR128(AND128(temp_1, _mm512_set1_epi32(0xffff0000)), AND128(temp_2, _mm512_set1_epi32(0x0000ffff)));

#define Round(Slice0, Slice1, Slice2, Slice3, Round_number)                                                                                 \
    Subcell(Slice0, Slice1, Slice2, Slice3);                                                                                                \
    Permute_bits(Slice3, Slice1, Slice2, Slice0);                                                                                           \
    Add_RK_RC(Slice3, Slice1, Slice2, Slice0, K[(1 + (3 * Round_number)) % 4], K[(3 + (3 * Round_number)) % 4], Round_Const[Round_number]); \
    Key_Update(Round_number);

#define GIFT(S0, S1, S2, S3)   \
    Round(S0, S1, S2, S3, 0);  \
    Round(S3, S1, S2, S0, 1);  \
    Round(S0, S1, S2, S3, 2);  \
    Round(S3, S1, S2, S0, 3);  \
    Round(S0, S1, S2, S3, 4);  \
    Round(S3, S1, S2, S0, 5);  \
    Round(S0, S1, S2, S3, 6);  \
    Round(S3, S1, S2, S0, 7);  \
    Round(S0, S1, S2, S3, 8);  \
    Round(S3, S1, S2, S0, 9);  \
    Round(S0, S1, S2, S3, 10); \
    Round(S3, S1, S2, S0, 11); \
    Round(S0, S1, S2, S3, 12); \
    Round(S3, S1, S2, S0, 13); \
    Round(S0, S1, S2, S3, 14); \
    Round(S3, S1, S2, S0, 15); \
    Round(S0, S1, S2, S3, 16); \
    Round(S3, S1, S2, S0, 17); \
    Round(S0, S1, S2, S3, 18); \
    Round(S3, S1, S2, S0, 19); \
    Round(S0, S1, S2, S3, 20); \
    Round(S3, S1, S2, S0, 21); \
    Round(S0, S1, S2, S3, 22); \
    Round(S3, S1, S2, S0, 23); \
    Round(S0, S1, S2, S3, 24); \
    Round(S3, S1, S2, S0, 25); \
    Round(S0, S1, S2, S3, 26); \
    Round(S3, S1, S2, S0, 27); \
    Round(S0, S1, S2, S3, 28); \
    Round(S3, S1, S2, S0, 29); \
    Round(S0, S1, S2, S3, 30); \
    Round(S3, S1, S2, S0, 31); \
    Round(S0, S1, S2, S3, 32); \
    Round(S3, S1, S2, S0, 33); \
    Round(S0, S1, S2, S3, 34); \
    Round(S3, S1, S2, S0, 35); \
    Round(S0, S1, S2, S3, 36); \
    Round(S3, S1, S2, S0, 37); \
    Round(S0, S1, S2, S3, 38); \
    Round(S3, S1, S2, S0, 39);

#define Shuffle_S0(a) _mm512_shuffle_epi8(a, shuffle_mask_S0);
#define Shuffle_S1(a) _mm512_shuffle_epi8(a, shuffle_mask_S1);
#define Shuffle_S2(a) _mm512_shuffle_epi8(a, shuffle_mask_S2);
#define Shuffle_S3(a) _mm512_shuffle_epi8(a, shuffle_mask_S3);

#define DeclareVar                                            \
    V256 S0, S1, S2, S3, temp, temp_1, temp_2;                \
    V256 K[4];                                                \
    V256 trmask1 = CONST128(maketrmask1);                     \
    V256 trmask2 = CONST128(maketrmask2);                     \
    V256 trmask3 = CONST128(maketrmask3);                     \
    V256 shuffle_mask = CONST128(make_shuffle_mask);          \
    V256 slice_shuffle_mask = CONST128(makesliceshufflemask); \
    V256 shuffle_mask_S0 = CONST128(make_shuffle_mask_S0);    \
    V256 shuffle_mask_S1 = CONST128(make_shuffle_mask_S1);    \
    V256 shuffle_mask_S2 = CONST128(make_shuffle_mask_S2);    \
    V256 shuffle_mask_S3 = CONST128(make_shuffle_mask_S3);

#define State2Var \
    S0 = LOAD128(State[0]), S1 = LOAD128(State[16]), S2 = LOAD128(State[16*2]), S3 = LOAD128(State[16*3]);

#define LoadKey \
    K[0] = LOAD128(Key[0]), K[1] = LOAD128(Key[16]), K[2] = LOAD128(Key[16*2]), K[3] = LOAD128(Key[16*3]);

/*Data for use input Key as such*/
uint8_t Round_Const[40] = {0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F,
                           0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B,
                           0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E,
                           0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A};
CGAL_ALIGN_16 uint32_t input[4] = {0x64786478, 0x64786478, 0x64786478, 0x64786478};
CGAL_ALIGN_16 uint32_t Key[16 * 4] = {0x474c5f29, 0x474c5f29, 0x474c5f29, 0x474c5f29, 0x474c5f29, 0x474c5f29, 0x474c5f29, 0x474c5f29,
				      0x474c5f29, 0x474c5f29, 0x474c5f29, 0x474c5f29, 0x474c5f29, 0x474c5f29, 0x474c5f29, 0x474c5f29,
                                      0x1f7b8aa6, 0x1f7b8aa6, 0x1f7b8aa6, 0x1f7b8aa6, 0x1f7b8aa6, 0x1f7b8aa6, 0x1f7b8aa6, 0x1f7b8aa6,
                                      0x1f7b8aa6, 0x1f7b8aa6, 0x1f7b8aa6, 0x1f7b8aa6, 0x1f7b8aa6, 0x1f7b8aa6, 0x1f7b8aa6, 0x1f7b8aa6,
                                      0x3138709b, 0x3138709b, 0x3138709b, 0x3138709b, 0x3138709b, 0x3138709b, 0x3138709b, 0x3138709b,
                                      0x3138709b, 0x3138709b, 0x3138709b, 0x3138709b, 0x3138709b, 0x3138709b, 0x3138709b, 0x3138709b,
                                      0xf8f1480e, 0xf8f1480e, 0xf8f1480e, 0xf8f1480e, 0xf8f1480e, 0xf8f1480e, 0xf8f1480e, 0xf8f1480e,
                                      0xf8f1480e, 0xf8f1480e, 0xf8f1480e, 0xf8f1480e, 0xf8f1480e, 0xf8f1480e, 0xf8f1480e, 0xf8f1480e};

CGAL_ALIGN_16 uint32_t State[16 * 4] = {0x3bce9899, 0x3bce9899, 0x3bce9899, 0x3bce9899, 0x3bce9899, 0x3bce9899, 0x3bce9899, 0x3bce9899,
					0x3bce9899, 0x3bce9899, 0x3bce9899, 0x3bce9899, 0x3bce9899, 0x3bce9899, 0x3bce9899, 0x3bce9899,
                                        0xb17fb330, 0xb17fb330, 0xb17fb330, 0xb17fb330, 0xb17fb330, 0xb17fb330, 0xb17fb330, 0xb17fb330,
                                        0xb17fb330, 0xb17fb330, 0xb17fb330, 0xb17fb330, 0xb17fb330, 0xb17fb330, 0xb17fb330, 0xb17fb330,
                                        0xfc130225, 0xfc130225, 0xfc130225, 0xfc130225, 0xfc130225, 0xfc130225, 0xfc130225, 0xfc130225,
                                        0xfc130225, 0xfc130225, 0xfc130225, 0xfc130225, 0xfc130225, 0xfc130225, 0xfc130225, 0xfc130225,
                                        0x566c194e, 0x566c194e, 0x566c194e, 0x566c194e, 0x566c194e, 0x566c194e, 0x566c194e, 0x566c194e,
                                        0x566c194e, 0x566c194e, 0x566c194e, 0x566c194e, 0x566c194e, 0x566c194e, 0x566c194e, 0x566c194e};
/*Mask for transpose operation and swapmove operation*/
uint16_t maketrmask1[8 * 4] = {0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842,
			       0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842, 0x0842};       // for first super diagonal
uint16_t maketrmask2[8 * 4] = {0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084,
			       0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084, 0x0084};       // for 2nd super diagonal
uint16_t maketrmask3[8 * 4] = {0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008,
			       0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008, 0x0008};       // for 3rd super diagonal
uint16_t make_shuffle_mask[8 * 4] = {0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0,
				     0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0}; // mask for swapmove used in interlacing shiffle applied to each slice

/*Subsequent epi shuffle masks*/
uint8_t makesliceshufflemask[16 * 4] = {0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15,
                                        0 + 16, 2 + 16, 1 + 16, 3 + 16, 4 + 16, 6 + 16, 5 + 16, 7 + 16, 8 + 16, 10 + 16, 9 + 16, 11 + 16, 12 + 16, 14 + 16, 13 + 16, 15 + 16,
                                        0 + (16*2), 2 + (16*2), 1 + (16*2), 3 + (16*2), 4 + (16*2), 6 + (16*2), 5 + (16*2), 7 + (16*2), 8 + (16*2), 10 + (16*2), 9 + (16*2), 11 + (16*2), 12 + (16*2), 14 + (16*2), 13 + (16*2), 15 + (16*2),
                                        0 + (16*3), 2 + (16*3), 1 + (16*3), 3 + (16*3), 4 + (16*3), 6 + (16*3), 5 + (16*3), 7 + (16*3), 8 + (16*3), 10 + (16*3), 9 + (16*3), 11 + (16*3), 12 + (16*3), 14 + (16*3), 13 + (16*3), 15 + (16*3)};

uint8_t make_shuffle_mask_S0[16 * 4] = {0, 3, 2, 1, 4, 7, 6, 5, 8, 11, 10, 9, 12, 15, 14, 13,
                                        0 + 16, 3 + 16, 2 + 16, 1 + 16, 4 + 16, 7 + 16, 6 + 16, 5 + 16, 8 + 16, 11 + 16, 10 + 16, 9 + 16, 12 + 16, 15 + 16, 14 + 16, 13 + 16,
                                        0 + (16*2), 3 + (16*2), 2 + (16*2), 1 + (16*2), 4 + (16*2), 7 + (16*2), 6 + (16*2), 5 + (16*2), 8 + (16*2), 11 + (16*2), 10 + (16*2), 9 + (16*2), 12 + (16*2), 15 + (16*2), 14 + (16*2), 13 + (16*2),
                                        0 + (16*3), 3 + (16*3), 2 + (16*3), 1 + (16*3), 4 + (16*3), 7 + (16*3), 6 + (16*3), 5 + (16*3), 8 + (16*3), 11 + (16*3), 10 + (16*3), 9 + (16*3), 12 + (16*3), 15 + (16*3), 14 + (16*3), 13 + (16*3)};
                                        
                                        
uint8_t make_shuffle_mask_S1[16 * 4] = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
                                        1 + 16, 0 + 16, 3 + 16, 2 + 16, 5 + 16, 4 + 16, 7 + 16, 6 + 16, 9 + 16, 8 + 16, 11 + 16, 10 + 16, 13 + 16, 12 + 16, 15 + 16, 14 + 16,
                                        1 + (16*2), 0 + (16*2), 3 + (16*2), 2 + (16*2), 5 + (16*2), 4 + (16*2), 7 + (16*2), 6 + (16*2), 9 + (16*2), 8 + (16*2), 11 + (16*2), 10 + (16*2), 13 + (16*2), 12 + (16*2), 15 + (16*2), 14 + (16*2),
                                        1 + (16*3), 0 + (16*3), 3 + (16*3), 2 + (16*3), 5 + (16*3), 4 + (16*3), 7 + (16*3), 6 + (16*3), 9 + (16*3), 8 + (16*3), 11 + (16*3), 10 + (16*3), 13 + (16*3), 12 + (16*3), 15 + (16*3), 14 + (16*3)};
uint8_t make_shuffle_mask_S2[16 * 4] = {2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15,
                                        2 + 16, 1 + 16, 0 + 16, 3 + 16, 6 + 16, 5 + 16, 4 + 16, 7 + 16, 10 + 16, 9 + 16, 8 + 16, 11 + 16, 14 + 16, 13 + 16, 12 + 16, 15 + 16,
                                        2+(16*2), 1+(16*2), 0+(16*2), 3+(16*2), 6+(16*2), 5+(16*2), 4+(16*2), 7+(16*2), 10+(16*2), 9+(16*2), 8+(16*2), 11+(16*2), 14+(16*2), 13+(16*2), 12+(16*2), 15+(16*2),
                                        2+(16*3), 1+(16*3), 0+(16*3), 3+(16*3), 6+(16*3), 5+(16*3), 4+(16*3), 7+(16*3), 10+(16*3), 9+(16*3), 8+(16*3), 11+(16*3), 14+(16*3), 13+(16*3), 12+(16*3), 15+(16*3)};
uint8_t make_shuffle_mask_S3[16 * 4] = {3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                                        3 + 16, 2 + 16, 1 + 16, 0 + 16, 7 + 16, 6 + 16, 5 + 16, 4 + 16, 11 + 16, 10 + 16, 9 + 16, 8 + 16, 15 + 16, 14 + 16, 13 + 16, 12 + 16,
                                        3+(16*2), 2+(16*2), 1+(16*2), 0+(16*2), 7+(16*2), 6+(16*2), 5+(16*2), 4+(16*2), 11+(16*2), 10+(16*2), 9+(16*2), 8+(16*2), 15+(16*2), 14+(16*2), 13+(16*2), 12+(16*2),
                                        3+(16*3), 2+(16*3), 1+(16*3), 0+(16*3), 7+(16*3), 6+(16*3), 5+(16*3), 4+(16*3), 11+(16*3), 10+(16*3), 9+(16*3), 8+(16*3), 15+(16*3), 14+(16*3), 13+(16*3), 12+(16*3)};

uint8_t make_shuffle_mask_all_slices[16 * 4] = {0, 3, 2, 1, 5, 4, 7, 6, 10, 9, 8, 11, 15, 14, 13, 12,
                                                0 + 16, 3 + 16, 2 + 16, 1 + 16, 5 + 16, 4 + 16, 7 + 16, 6 + 16, 10 + 16, 9 + 16, 8 + 16, 11 + 16, 15 + 16, 14 + 16, 13 + 16, 12 + 16,
                                                0+(16*2), 3+(16*2), 2+(16*2), 1+(16*2), 5+(16*2), 4+(16*2), 7+(16*2), 6+(16*2), 10+(16*2), 9+(16*2), 8+(16*2), 11+(16*2), 15+(16*2), 14+(16*2), 13+(16*2), 12+(16*2),
                                                0+(16*3), 3+(16*3), 2+(16*3), 1+(16*3), 5+(16*3), 4+(16*3), 7+(16*3), 6+(16*3), 10+(16*3), 9+(16*3), 8+(16*3), 11+(16*3), 15+(16*3), 14+(16*3), 13+(16*3), 12+(16*3)};

void main()
{
    CGAL_ALIGN_16 uint32_t buf[4];
    DeclareVar;
    LoadKey;
    State2Var;
    printf("initial\n");
     dump_state();

    //GIFT(S0, S1, S2, S3);
    MEASURE(GIFT(S0, S1, S2, S3));
    printf("%f,%f,%f",RDTSC_clk_min ,RDTSC_clk_median,RDTSC_clk_max );
    
     printf("Final State\n");
    dump_state();
     printf("Final Key\n");
    dump_key();
}
