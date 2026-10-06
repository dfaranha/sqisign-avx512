#include "sqi_e0_basis.h"

// Basis of E0[2^324] for p324_3, Montgomery form with R = 2^384
// (from the SQIsign v3 reference, src/precomp/ref/p324_3/e0_basis.c)
const fp2_t BASIS_E0_PX = {
    { .arr = { 0x15429ebdd6acc6c8, 0x1bb9d347e507e96f, 0x345b16e8f3d297c5, 0x216fa909d6954b0f, 0x34f99a0a51ded6ca, 0xe } },
    { .arr = { 0xf8d03b133d9ee6ef, 0xdd0cdd882713e200, 0x7aab08998f08531b, 0x13785435a9b9f681, 0x20f45f7e1ba8367d, 0x9 } }
};

const fp2_t BASIS_E0_QX = {
    { .arr = { 0x92aa928d45684b84, 0x6287060c7e5e212c, 0x1818aad55dfc3fb0, 0x2c34acd0ee182e1, 0x83223a491fa840d9, 0x21 } },
    { .arr = { 0x934ae51618b44717, 0x543fa24873173bce, 0x43046078cb60d862, 0x3fa561a4701041fe, 0xfb2a0dfe18c42e16, 0x13 } }
};
