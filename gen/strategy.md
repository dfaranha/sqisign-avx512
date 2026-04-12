Requirements on p:

The script only designed for cryptographically sized primes, so we assume
p_bits >= 51 * 4

We require c to be small, specifically that  c_bits <= min_limb_len.
We have limb_len = ceil(pbits/ceil(pbits/51))
For a given num_limbs, this is minimized when pbits = 51*(num_limbs-1) + 1,
which is the smallest p_bits for which the denominator increases by 1, to num_limbs.
min_limb_len = ceil(pbits/num_limbs) = ceil(51 - 50/num_limbs) = 51 - floor(50/num_limbs) 
  = 52 - (floor(50/num_limbs) + 1) = 52 - ceil(51/num_limbs)
With num_limbs >= 4, we get min_limb_len >= 39

For the Montgomery reduction, we rely on the fact that -(p^-1) mod R = c*2^e + 1
(c * 2^e - 1) * (c*2^e + 1) = c^2 * 2^(2e) - 1.
Hence this is true iff 2e >= num_limbs_R * limb_len
num_limbs_R * limb_len <= (num_limbs+1) * limb_len <= (num_limbs+1) * limb_len + last_limb_len
  = p_bits + 2 * limb_len = e + c_bits + 2 * limb_len <= e + 3* limb_len
Hence it is sufficient that 3*limb_len <= e.
Since e = p_bits - c_bits >= 51 * 4 - c_bits >= 51 * 4 - 39 > 51*3, this is always satisfied.

======================================================================================
At the moment we have simplifying assumption c_bits + (e mod limb_len) < last_limb_len
If this assumption dos not hold, mp_add_p_sub must be changed (and maybe parts of rdc)
======================================================================================

Considerations for Fp2 lazy multiplication:
We compute     
re = a_re * b_re - a_im * b_im
im = (a_re + a_im) * (b_re + b_im) - a_re * b_re - a_im * b_im = a_re * b_im + b_re * b_im
Over the integers, re can be negative, while im is always positive.
If we reduce mod p:  re in [-p^2, p^2]   and im in [0, 2p^2]
If we reduce mod 2p: re in [-4p^2, 4p^2] and im in [0, 8p^2]

1. Fp multiplication strategy
With Fp Karatsuba, we compute (a_lo+a_hi)*(b_lo+b_hi), so input limbs must be 51 bits. This is why we never go above 51 bits for limb_len.
With Fp schoolbook, the input can have 52 bit limbs.
Depending on the Fp2 multiplication strategy, we need to ensure that we don't call Fp mul with too large limbs.

Fp2 schoolbook with Fp schoolbook (2x2x2), Fp2 Karatsuba with Fp schoolbook (4x1x2):
Multiplication:
  Karatsuba increases the limbs to limb_len + 1 <= 52 bits, which is below the AVX mul threshold. 
  As long as last_limb_len < limb_len or we reduce mod p, the last limb is also at most 52 bits.
Squaring:
  Uses mp_add_p_sub. When limb_len = 51 and we reduce mod 2p, it adds 2*limb_mask to the first limb, 
  so the result grows to 53 bits. But with carry propagation, it works as long as last_limb_len <= 50, otherwise we have to switch to mod p.

Fp2 schoolbook with Fp Karatsuba (4x2x1) 
Multiplication: Does mul before adds, so same applies as previous.
Squaring: Adds before mul. For 51 bit limbs, need to carry. When last_limb_len = 51, carry is not enough, we need to reduce everything.   

Fp2 Karatsuba with Fp Karatsuba (8x1x1):
Multiplication:
  The standard lazy approach requires limb_len <= 50. Then a_re + a_im is 51 bits, so Fp Karatsuba works.
  When limb_len = 51, we can propogate carries for a_re + a_im. The top limb grows by 1 bit, and the others are 51 bits.
  If last_limb_len <= 50 and mod p (or last_limb_len <= 49 and mod 2p): last_limb + carry is still at most 51 bits.
  Otherwise: last_limb + carry can be 52 bits. Then we need a conditional subtraction by p and another propagation. 
  Now potentially the computation of im can be negative, etc... We postpone the discussion of what to do here until the end.
Squaring:
  For the squaring, we do mp_add_p_sub. It adds 2*limb_mask to the first limbs, so that for limb_len = 50, we get that the first limb is 52 bits.
  So we need to propagate the carry at limb_len >= 50. It works mod 2p as long as last_limb_len <= 48 and mod p as long as last_limb_len <= 49

2. Montgomery reduction requires input in [0, p*R).
There are two choices for the Montgomery constant R: The standard R_st = 2^(num_limbs * limb_len) or R_ext = 2^((num_limbs+1)*limb_len)
Cost of using R_ext: 
  One extra limb in the s computation and one extra mult when computing s * top.
  Importantly, the cost is constant, it doesn't depend on num_limbs or limb_len. Allow pipelining.
Benefits of using R_ext: Potentially easier to stay within [0, p*R) in lazy reduction.
There is also the choice of whether to reduce mod p or to reduce mod 2p (i.e. whether to skip the last conditional subtraction by p in the red).
We analyze each of the four cases, before choosing a strategy.

R_ext and mod 2p:
8p^2 < 8p * R_st < p * R_ext, so im is in [0, p * R_ext). To make re positive, we can always add p*R_st to it.
p * R_st - p^2 > p * R_st - p * R_st = 0 as p < R_st.
p * R_st + p^2 < 2 * p * R_st < p * R_ext.
Thus, re + p * R_st in [0, p*R_ext)

R_ext and mod p: The same argument holds. Again we must always add p*R to re.

R_st and mod p:
Two ways to handle re being negative:
  a) Conditionally handle add p*R_st
  b) Fix it at the very end of the Montgomery reduction. If the result is in (-p*R, 0) before reduction, after it is in (-p, p).
     This requries -p*R_st < -p^2, which always holds.
If last_limb_len < limb_len: 2p < R_st and im < 2p^2 < p*R_st.
ISSUE wih squaring. mp_add_p_sub increases beyond bound, recommend switching to R_ext.
If last_limb_len = limb_len: 2p > R_st. Observe that since p < R_st, 2p - R_st < p
  Then by conditionally subtracting im by p*R_st, it's new upper bound is 
  2p^2 - p * R_st = p*(2p - R_st) < p^2 < p*R_st.

R_st and mod 2p:
Observe that 2^(limb_len - last_limb_len)*p < R_st.
If last_limb_len <= limb_len - 3: im is [0, p*R_st). Both approach a) and b) for handling re works in this case, 
  the latter because -p*R_st <= -8p^2 < -4p^2 < re.
If last_limb_len = limb_len -2, im is outside of the range. It is sufficient to conditionally subtract im by p*R, but at his point we migh as well reduce by p.
So with R_st and last_limb_len >= limb_len-2, we recommend reducing mod p.

Our strategy:
When limb_len <= 50, last_limb_len <= 49:
  Fp2 Karatsuba: Lazy reduction
  R and mod: We pick R_ext and to reduce mod 2p. With R_st, we need to conditionally add p*R_st to re,
      which costs a sign check, a subtraction and a num_limbs-long serial carry propagation. This is more expensive than the cost of having R_ext.

limb_len = last_limb_len = 50:
  Fp2 Karatsuba: Lazy reduction
  R and mod: We pick R_ext, but have to reduce mod p. Otherwise the last limb is 51 bits, so a_re + a_im can be 52 bits.

== From experimental evidence, R_st and mod 2p is fastest for lvl1 ==

See the script for the rest of the strategy.....

      

