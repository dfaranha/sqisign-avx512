#include "sqi_e0_basis.h"

// Basis of E0[2^664] for p664_17, Montgomery form with R = 2^704
// (from the SQIsign v3 reference, src/precomp/ref/p664_17/e0_basis.c)
const fp2_t BASIS_E0_PX = {
    { .arr = { 0x9ee7b1cddd3559f0, 0xc12d626b16e7f831, 0x6e8a9c37304471ac, 0x3a91511d582f6788, 0x2af77787c35d2ebe, 0xd2a2d68eddeb735,
               0xb8cdde3aa33c88e0, 0x6a413264423297e7, 0xf2929004b9a981a9, 0x3a8a8b56e560ae9c, 0x10612794 } },
    { .arr = { 0x9674481708b16d15, 0x7de13a9378bd9a1a, 0xf4f9dbb17ec86c32, 0xbfb000d851c583ab, 0xc1b3af1fb0f09461, 0xd380edebc28eda15,
               0xe094ccedcd2b2497, 0xcbb4874482e31a51, 0x2c37af3de3fb6672, 0x16ffcbfc980ad048, 0x597a4ed } }
};

const fp2_t BASIS_E0_QX = {
    { .arr = { 0xf9f502a4264444c7, 0x9de2b170711d43ae, 0xdd0743dadf5fadff, 0xa0cd4480412a9bb4, 0x2a362b94c8508596, 0x4b1904c2df414af3,
               0xfe3982b7659ec94, 0x2cbf157bf4c645b9, 0x9e29887df0952b37, 0x86c29bc4a96070e3, 0x3532a6a } },
    { .arr = { 0x66bf5a7add5ed882, 0xe45221e4b95cf21f, 0x5d8cfe2aabb831d0, 0xd3071a7beb935b78, 0xafa5d3597c154ce8, 0xfa623421e7ced432,
               0xa495fdb6e1c5c261, 0x2aa681bbc2376c2, 0x9c5214fb65e0f597, 0xa985d1b959b9834a, 0x525e053 } }
};
