---
layout: post
title:  "Optimizing Adler-32 with RVV: Landing in zlib-ng and Chromium"
date:   2023-04-01 15:22:12 +0800
categories: [RISCV]
author: Alex Chiang
---

## What is Adler-32? ##

Adler-32 is a checksum algorithm which was invented by Mark Adler in 1995. A checksum algorithm is a function that takes an input (or 'message') and returns a fixed-size string of bytes, typically in the form of a hash value. The output is intended to uniquely represent the input data; even a small change to the input should produce such a drastic change in output that the new hash value appears uncorrelated with the old hash value.

Adler-32 is used in many software applications as a quick way to detect errors that may have been introduced into data during its transmission or storage. It's widely used in data transmission standards, such as in the zlib compression library, and for error checking in the software installation process.

Although Adler-32 is quick and easy to compute, it is not as reliable as other checksum methods like CRC32 for detecting errors, particularly in larger data sets, due to its simplicity. Nonetheless, it offers a reasonable compromise between speed and reliability for many applications.

### The formula ###

$$
\begin{array}{l}
  A &= & 1 + D_{0} + D_{1} + D_{2} + ... + D_{n} \mod 65521 \\
  B &= & 1 + D_{0} +                                        \\
    &\;& 1 + D_{0} + D_{1} +                                \\
    &\;& 1 + D_{0} + D_{1} + D_{2} +                        \\
    &\;& ... +                                              \\
    &\;& 1 + D_{0} + D_{1} + D_{2} + ... + D_{n} \mod 65521 \\
  Adler32 &=& B × 65536 + A
\end{array}
$$

where $$D$$ is the input data bytes for which the checksum is to be calculated, and $$n + 1$$ is the length of $$D$$.

The simple c code:

```c
#define BASE 65521

uint32_t adler32(uint32_t adler, const uint8_t *buf, size_t len) {
  uint32_t sum2;
  /* split Adler-32 into component sums */
  sum2 = (adler >> 16) & 0xffff;
  adler &= 0xffff;

  while (len--) {
    adler += *buf++;
    sum2 += adler;
    adler %= BASE;
    sum2 %= BASE;
  }

  return adler | (sum2 << 16);
}
```

## Adler32 in RISC-V Arch ##

Assume that we process $$D_{x}$$ ~ $$D_{y}$$ for each iteration, where $$x$$ and $$y$$ both are the indexes of the input data bytes, and $$(y - x + 1) = vl$$.

$$
\begin{array}{l}
  A &= & A_{last} + D_{x} + D_{x+1} + D_{x+2} + ... + D_{y} \mod 65521 \\
  B &= & A_{last} + D_{x} +                                            \\
    &\;& A_{last} + D_{x} + D_{x+1} +                                  \\
    &\;& A_{last} + D_{x} + D_{x+1} + D_{x+2} +                        \\
    &\;& A_{last} + ...   +                                            \\
    &\;& A_{last} + D_{x} + D_{x+1} + D_{x+2} + ... + D_{y} \mod 65521 \\
    &= & vl * A_{last} + (vl)D_{x} + (vl-1)D_{x+1} + (vl-2)D_{x+2} + ... + D_{y} \mod 65521
\end{array}
$$

$$A_{last}$$ is the $$A$$ value from the last iteration.

However, modulo for each iteration is inefficient, we could move modulo operation to the last step. Such that,

$$
\begin{array}{l}
  A &= & A_{last} + D_{x} + D_{x+1} + D_{x+2} + ... + D_{y} \\
  B &= & A_{last} + D_{x} +                                 \\
    &\;& A_{last} + D_{x} + D_{x+1} +                       \\
    &\;& A_{last} + D_{x} + D_{x+1} + D_{x+2} +             \\
    &\;& A_{last} + ...   +                                 \\
    &\;& A_{last} + D_{x} + D_{x+1} + D_{x+2} + ... + D_{y} \\
    &= & vl * A_{last} + (vl)D_{x} + (vl-1)D_{x+1} + (vl-2)D_{x+2} + ... + D_{y} \\
\end{array}
$$

For example, assume that we progress 16 elements and set `vl` 4:

```txt
A =
// iteration 1
D0 + D1 + D2 + D3 +      => A_iteration_1
// iteration 2
D4 + D5 + D6 + D7 +      => A_iteration_2
// iteration 3
D8 + D9 + D10 + D11 +    => A_iteration_3
// iteration 4
D12 + D13 + D14 + D15    => A_iteration_4

B =
// iteration 1
D0 +
D0 + D1 +
D0 + D1 + D2 +
D0 + D1 + D2 + D3 +
// iteration 2
D0 + D1 + D2 + D3 + D4 +
D0 + D1 + D2 + D3 + D4 + D5 +
D0 + D1 + D2 + D3 + D4 + D5 + D6 +
D0 + D1 + D2 + D3 + D4 + D5 + D6 + D7 +
// iteration 3
D0 + D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 +
D0 + D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 +
D0 + D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 +
D0 + D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 + D11 +
// iteration 4
D0 + D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 + D11 + D12 +
D0 + D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 + D11 + D12 + D13 +
D0 + D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 + D11 + D12 + D13 + D14 +
D0 + D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 + D11 + D12 + D13 + D14 + D15 
```

And then, We reuse the $$A$$ from the last iteration when we accumulate data bytes to $$B$$:

```txt
B =
// iteration 1
D0 +
D0 + D1 +
D0 + D1 + D2 +
D0 + D1 + D2 + D3 +                       => we call it the triangle part below
// iteration 2
A_iteration_1 + D4 +
A_iteration_1 + D4 + D5 +
A_iteration_1 + D4 + D5 + D6 +
A_iteration_1 + D4 + D5 + D6 + D7 +
// iteration 3
A_iteration_2 + D8 +
A_iteration_2 + D8 + D9 +
A_iteration_2 + D8 + D9 + D10 +
A_iteration_2 + D8 + D9 + D10 + D11 +
// iteration 4
A_iteration_3 + D12 +
A_iteration_3 + D12 + D13 +
A_iteration_3 + D12 + D13 + D14 +
A_iteration_3 + D12 + D13 + D14 + D15 
```

Thus, we get the `version 1` codes:
```c
uint32_t adler32_rvv(uint32_t adler, const uint8_t *buf, size_t len) {
  /* split Adler-32 into component sums */
  uint32_t sum2 = (adler >> 16) & 0xffff;
  adler &= 0xffff;

  size_t vl = __riscv_vsetvl_e8m1(len);
  size_t l = len;

  /* we accumulate the input 8-bit value, and to prevent overflow, we use 32-bit accumulator */
  vuint32m4_t v_adler = __riscv_vmv_v_x_u32m4(0, vl);
  vuint32m4_t v_sum2 = __riscv_vmv_v_x_u32m4(0, vl);

  /* create an array of [vl, vl - 1, vl - 2, ..., 1] */
  vuint16m2_t v_seq = __riscv_vid_v_u16m2(vl);
  vuint16m2_t v_vl_arr = __riscv_vmv_v_x_u16m2(vl, vl);
  vuint16m2_t v_rev_seq = __riscv_vsub_vv_u16m2(v_vl_arr, v_seq, vl);

  while (l >= vl) {
    vuint8m1_t v_buf = __riscv_vle8_v_u8m1(buf, vl);
    v_sum2 = __riscv_vmacc_vx_u32m4(v_sum2, vl, v_adler, vl);
    vuint16m2_t v_buf16 = __riscv_vzext_vf2_u16m2(v_buf, vl);

    /* calculate triangle part */
    v_sum2 = __riscv_vwmaccu_vv_u32m4(v_sum2, v_buf16, v_rev_seq, vl);
    v_adler = __riscv_vwaddu_wv_u32m4(v_adler, v_buf16, vl);

    buf += vl;
    l -= vl;
  }

  vuint32m1_t v_sum2_tmp = __riscv_vmv_s_x_u32m1(0, vl);
  v_sum2_tmp = __riscv_vredsum(v_sum2, v_sum2_tmp, vl);
  uint32_t sum2_tmp = __riscv_vmv_x_s_u32m1_u32(v_sum2_tmp);

  /* don't forget the adler value base on the input one */
  sum2 += (sum2_tmp + adler * (len - l));

  vuint32m1_t v_adler_sum = __riscv_vmv_s_x_u32m1(0, vl);
  v_adler_sum = __riscv_vredsum(v_adler, v_adler_sum, vl);
  uint32_t adler_sum = __riscv_vmv_x_s_u32m1_u32(v_adler_sum);

  adler += adler_sum;

  /* handling remaining data using scalar codes */
  while (l--) {
    adler += *buf++; 
    sum2 += adler;
  }

  /* do modulo at last step */
  sum2 %= BASE;
  adler %= BASE;

  return adler | (sum2 << 16);
}
```

It is a straightforward implementation, but we still have some room for optimization of the algorithm.

We add an 8-bit vector into a 32-bit accumulator, which is inefficient, because we have to extend an 8-bit data to 32-bit with zeros, and then vector addition is possible.

We could add an 8-bit vector into a 16-bit accumulator, because using a 16-bit accumulator is cheaper, and a vector wide-addition instruction for RISC-V(`vwaddu`) would be useful here. Then, add the value from the 16-bit accumulator to the 32-bit one before an overflow occurs.

Let's see the code below.
```c
// create a 16-bit accumulator
// create a 32-bit accumulator

/* We split the input data into block_size chunks, and block_size should be <= 256,
 * because an overflow for 16-bit would occur when block_size > 256 (255 * 256 <= UINT16_MAX).
 * This assumes that all input data bytes are all 255 (UINT8_MAX) for worstest scenario.
 */
size_t block_size = 256;
while (len >= block_size) {
  // clean 16-bit accumulator
  size_t subprob = block_size;
  while (subprob > 0) {
    // vector add 8-bit data into 16-bit accumulator
    subprob -= vl;
  }
  // accumulate 16-bit accumulator value into the 32-bit accumulator at the last iteration.
  len -= block_size;
}

/* now, the remaining data length <= 256, we could use 16-bit accu safely */

// clean 16-bit accumulator
while (len >= vl) {
  // vector add 8-bit data into 16-bit accumulator safely
  len -= vl;
}

/* handling remaining part using scalar code */
while (len > 0) {
  // ...
}
```

Such that, we have reduced the number of times we use a 32-bit accumulator to $$\frac{1}{256}$$ of its original usage.

We also can move the multiplication operation to the last step because the integral have distributive property ($$AC + BC = (A + B)C$$). Continuing with the above example:

```txt
B =
  // rectangle part
  4(A_iteration_1 + A_iteration_2 + A_iteration_3)
  // triangle part
  + 4(D0 + D4 + D8 + D12)
  + 3(D1 + D5 + D9 + D13)
  + 2(D2 + D6 + D10 + D14)
  + 1(D3 + D7 + D11 + D15)
```

Thus, we get `version 2` codes:

```c
uint32_t adler32_rvv(uint32_t adler, const uint8_t *buf, size_t len) {
  /* split Adler-32 into component sums */
  uint32_t sum2 = (adler >> 16) & 0xffff;
  adler &= 0xffff;

  size_t left = len;
  size_t vl = __riscv_vsetvlmax_e8m1();
  vl = vl > 256 ? 256 : vl;
  vuint32m4_t v_buf32_accu = __riscv_vmv_v_x_u32m4(0, vl);
  vuint32m4_t v_adler32_prev_accu = __riscv_vmv_v_x_u32m4(0, vl);
  vuint16m2_t v_buf16_accu;

  /*
   * We accumulate 8-bit data, and to prevent overflow, we have to use a 32-bit accumulator.
   * However, adding 8-bit data into a 32-bit accumulator isn't efficient. We use 16-bit & 32-bit
   * accumulators to boost performance.
   *
   * The block_size is the largest multiple of vl that <= 256, because overflow would occur when
   * vl > 256 (255 * 256 <= UINT16_MAX).
   * 
   * We accumulate 8-bit data into a 16-bit accumulator and then
   * move the data into the 32-bit accumulator at the last iteration.
   */
  size_t block_size = (256 / vl) * vl;
  while (left >= block_size) {
    v_buf16_accu = __riscv_vmv_v_x_u16m2(0, vl);
    size_t subprob = block_size;
    while (subprob > 0) {
      vuint8m1_t v_buf8 = __riscv_vle8_v_u8m1(buf, vl);
      v_adler32_prev_accu = __riscv_vwaddu_wv_u32m4(v_adler32_prev_accu, v_buf16_accu, vl);
      v_buf16_accu = __riscv_vwaddu_wv_u16m2(v_buf16_accu, v_buf8, vl);
      buf += vl;
      subprob -= vl;
    }
    /* calculate rectangle part */
    v_adler32_prev_accu = __riscv_vmacc_vx_u32m4(v_adler32_prev_accu, block_size / vl, v_buf32_accu, vl);
    v_buf32_accu = __riscv_vwaddu_wv_u32m4(v_buf32_accu, v_buf16_accu, vl);
    left -= block_size;
  }
  /* the left len <= 256 now, we can use 16-bit accum safetly */
  v_buf16_accu = __riscv_vmv_v_x_u16m2(0, vl);
  size_t res = left;
  while (left >= vl) {
    vuint8m1_t v_buf8 = __riscv_vle8_v_u8m1(buf, vl);
    v_adler32_prev_accu = __riscv_vwaddu_wv_u32m4(v_adler32_prev_accu, v_buf16_accu, vl);
    v_buf16_accu = __riscv_vwaddu_wv_u16m2(v_buf16_accu, v_buf8, vl);
    buf += vl;
    left -= vl;
  }
  /* calculate rectangle part */
  v_adler32_prev_accu = __riscv_vmacc_vx_u32m4(v_adler32_prev_accu, res / vl, v_buf32_accu, vl);
  v_buf32_accu = __riscv_vwaddu_wv_u32m4(v_buf32_accu, v_buf16_accu, vl);

  /* create an array of [vl, vl - 1, vl - 2, ..., 1] */
  vuint32m4_t v_seq = __riscv_vid_v_u32m4(vl);
  vuint32m4_t v_rev_seq = __riscv_vrsub_vx_u32m4(v_seq, vl, vl);

  /* calculate triangle part */
  vuint32m4_t v_sum32_accu = __riscv_vmul_vv_u32m4(v_buf32_accu, v_rev_seq, vl); 

  v_sum32_accu = __riscv_vadd_vv_u32m4(v_sum32_accu, __riscv_vmul_vx_u32m4(v_adler32_prev_accu, vl, vl), vl);

  vuint32m1_t v_sum2_sum = __riscv_vmv_s_x_u32m1(0, vl);
  v_sum2_sum = __riscv_vredsum_vs_u32m4_u32m1(v_sum32_accu, v_sum2_sum, vl);
  uint32_t sum2_sum = __riscv_vmv_x_s_u32m1_u32(v_sum2_sum);

  /* don't forget the adler value base on the input one */
  sum2 += (sum2_sum + adler * (len - left));

  vuint32m1_t v_adler_sum = __riscv_vmv_s_x_u32m1(0, vl);
  v_adler_sum = __riscv_vredsum_vs_u32m4_u32m1(v_buf32_accu, v_adler_sum, vl);
  uint32_t adler_sum = __riscv_vmv_x_s_u32m1_u32(v_adler_sum);

  adler += adler_sum;
 
  /* hadling remaining data using scalar codes */
  while (left--) {
    adler += *buf++;
    sum2 += adler;
  }

  sum2 %= BASE;
  adler %= BASE;

  return adler | (sum2 << 16);
}
```

Finally, we have to consider that doing modulo once at the last step may be a risk.
If we handle a very long input data, we cannot ensure that an overflow won't occur for the 32-bit accumulator during computing.

Assume we always progress $n$ elements from the last modulo, in the worst-case scenario, the current maximum value of `B` is $$255 * \frac{n(n + 1)}{2}$$, and `A` is $$n * (\text{BASE} - 1)$$. And don't forget that we still have value after doing the last modulo, the $$(\text{BASE} - 1)$$ would be the maximum value possibility.

So, we could find the maximum $$n$$ that $$255n(n+1)/2 + (n+1)(\text{BASE}-1) <= 2^32-1 \Rightarrow n = 5552$$ (it's also defined at `zlib-ng` as `NMAX`), so we have to do modulo once for each block of `NMAX` size.

Thus, we get the `version 3` codes:

```c
uint32_t adler32_rvv(uint32_t adler, const uint8_t *buf, size_t len) {
  /* split Adler-32 into component sums */
  uint32_t sum2 = (adler >> 16) & 0xffff;
  adler &= 0xffff;

  size_t left = len;
  size_t vl = __riscv_vsetvlmax_e8m1();
  vl = vl > 256 ? 256 : vl;
  vuint32m4_t v_buf32_accu = __riscv_vmv_v_x_u32m4(0, vl);
  vuint32m4_t v_adler32_prev_accu = __riscv_vmv_v_x_u32m4(0, vl);
  vuint16m2_t v_buf16_accu;

  /*
   * We accumulate 8-bit data, and to prevent overflow, we have to use a 32-bit accumulator.
   * However, adding 8-bit data into a 32-bit accumulator isn't efficient. We use 16-bit & 32-bit
   * accumulators to boost performance.
   *
   * The block_size is the largest multiple of vl that <= 256, because overflow would occur when
   * vl > 256 (255 * 256 <= UINT16_MAX).
   * 
   * We accumulate 8-bit data into a 16-bit accumulator and then
   * move the data into the 32-bit accumulator at the last iteration.
   */
  size_t block_size = (256 / vl) * vl;
  size_t nmax_limit = (NMAX / block_size);
  size_t cnt = 0;
  while (left >= block_size) {
    v_buf16_accu = __riscv_vmv_v_x_u16m2(0, vl);
    size_t subprob = block_size;
    while (subprob > 0) {
      vuint8m1_t v_buf8 = __riscv_vle8_v_u8m1(buf, vl);
      v_adler32_prev_accu = __riscv_vwaddu_wv_u32m4(v_adler32_prev_accu, v_buf16_accu, vl);
      v_buf16_accu = __riscv_vwaddu_wv_u16m2(v_buf16_accu, v_buf8, vl);
      buf += vl;
      subprob -= vl;
    }
    /* calculate rectangle part */
    v_adler32_prev_accu = __riscv_vmacc_vx_u32m4(v_adler32_prev_accu, block_size / vl, v_buf32_accu, vl);
    v_buf32_accu = __riscv_vwaddu_wv_u32m4(v_buf32_accu, v_buf16_accu, vl);
    left -= block_size;
    /* do modulo once each block of NMAX size */
    if (++cnt >= nmax_limit) {
      v_adler32_prev_accu = __riscv_vremu_vx_u32m4(v_adler32_prev_accu, BASE, vl);
      cnt = 0;
    }
  }
  /* the left len <= 256 now, we can use 16-bit accum safetly */
  v_buf16_accu = __riscv_vmv_v_x_u16m2(0, vl);
  size_t res = left;
  while (left >= vl) {
    vuint8m1_t v_buf8 = __riscv_vle8_v_u8m1(buf, vl);
    v_adler32_prev_accu = __riscv_vwaddu_wv_u32m4(v_adler32_prev_accu, v_buf16_accu, vl);
    v_buf16_accu = __riscv_vwaddu_wv_u16m2(v_buf16_accu, v_buf8, vl);
    buf += vl;
    left -= vl;
  }
  /* calculate rectangle part */
  v_adler32_prev_accu = __riscv_vmacc_vx_u32m4(v_adler32_prev_accu, res / vl, v_buf32_accu, vl);
  /* do modulo once again to prevent overflow */
  v_adler32_prev_accu = __riscv_vremu_vx_u32m4(v_adler32_prev_accu, BASE, vl);
  v_buf32_accu = __riscv_vwaddu_wv_u32m4(v_buf32_accu, v_buf16_accu, vl);

  /* create an array of [vl, vl - 1, vl - 2, ..., 1] */
  vuint32m4_t v_seq = __riscv_vid_v_u32m4(vl);
  vuint32m4_t v_rev_seq = __riscv_vrsub_vx_u32m4(v_seq, vl, vl);

  /* calculate triangle part */
  vuint32m4_t v_sum32_accu = __riscv_vmul_vv_u32m4(v_buf32_accu, v_rev_seq, vl); 

  v_sum32_accu = __riscv_vadd_vv_u32m4(v_sum32_accu, __riscv_vmul_vx_u32m4(v_adler32_prev_accu, vl, vl), vl);

  vuint32m1_t v_sum2_sum = __riscv_vmv_s_x_u32m1(0, vl);
  v_sum2_sum = __riscv_vredsum_vs_u32m4_u32m1(v_sum32_accu, v_sum2_sum, vl);
  uint32_t sum2_sum = __riscv_vmv_x_s_u32m1_u32(v_sum2_sum);

  sum2 += (sum2_sum + adler * (len - left));

  vuint32m1_t v_adler_sum = __riscv_vmv_s_x_u32m1(0, vl);
  v_adler_sum = __riscv_vredsum_vs_u32m4_u32m1(v_buf32_accu, v_adler_sum, vl);
  uint32_t adler_sum = __riscv_vmv_x_s_u32m1_u32(v_adler_sum);

  adler += adler_sum;

  /* hadling remaining data using scalar codes */
  while (left--) {
    adler += *buf++;
    sum2 += adler;
  }

  sum2 %= BASE;
  adler %= BASE;

  return adler | (sum2 << 16);
}
```

## Summary ##

This is the first patch for Adler-32 RVV optimization to land in zlib-ng and Chromium, introduced by Alex Chiang from SiFive.

1. Efficient Use of 16-Bit and 32-Bit Accumulators
The patch uses both 16-bit and 32-bit accumulators to efficiently process 8-bit data and the result from the last iteration, preventing overflow and enhancing performance.

1. Efficient Modulo Handling
The patch introduces a more efficient way to handle overflow and perform modulo operations. Instead of performing modulo after each iteration, do nodulo only once for each block of `NMAX` size.


## References ##

- [Patch landed in zlib-ng](https://github.com/zlib-ng/zlib-ng/commit/6eed7416ed38a7740da77e86f2e5be5e7bce586d)
- [Patch landed in chromium](https://chromium.googlesource.com/chromium/src/+/c0e7820262df6b9e69252babe4ffc1cccc1af135%5E%21/)
