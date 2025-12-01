#include "theta_gluing.h"
#include "fp2_lvl3.h"

//TODO improve implementation, remove the need to unvectorize.
void gluing_compute_matrix_4x2x1(
    __m512i* g1, __m512i* h1, __m512i* g2, __m512i* h2, __m512i* pi,
    __m512i* x, __m512i* z, __m512i* u, __m512i* w, __m512i* det
)
{
    // TODO: Copy comment on how this comp is performed.

    __m512i z_sqr[8], a[8], b[8], c[8], d[8];
    fp2_t a_un[4], b_un[4], c_un[4], d_un[4], out_un[4];

    fp2_sqr_4x2x1(z_sqr, z);
    fp2_mul_4x2x1(d, u, z_sqr);
    fp2_mul_4x2x1(b, w, z_sqr);
    fp2_neg_4x2x1(b, b);
    fp2_mul_4x2x1(c, u, z);
    fp2_sub_4x2x1(c, c, det);
    fp2_mul_4x2x1(c, c, x);
    fp2_neg_4x2x1(a, d);
    // Observe that this doesn't depend on the previous computation:
    fp2_mul_4x2x1(pi, z, det);

    fp2_4x2x1_to_unvectorized(a_un, a);
    fp2_4x2x1_to_unvectorized(b_un, b);
    fp2_4x2x1_to_unvectorized(c_un, c);
    fp2_4x2x1_to_unvectorized(d_un, d);

    out_un[0] = a_un[0]; out_un[1] = b_un[0]; out_un[2] = c_un[0]; out_un[3] = d_un[0];
    fp2_unvectorized_to_4x2x1(g1, out_un);
    out_un[0] = a_un[1]; out_un[1] = b_un[1]; out_un[2] = c_un[1]; out_un[3] = d_un[1];
    fp2_unvectorized_to_4x2x1(h1, out_un);
    out_un[0] = a_un[2]; out_un[1] = b_un[2]; out_un[2] = c_un[2]; out_un[3] = d_un[2];
    fp2_unvectorized_to_4x2x1(g2, out_un);
    out_un[0] = a_un[3]; out_un[1] = b_un[3]; out_un[2] = c_un[3]; out_un[3] = d_un[3];
    fp2_unvectorized_to_4x2x1(h2, out_un);
}