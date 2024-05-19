---
layout: post
title:  "Optimizing Adler-32 with RVV: A Deep Dive into Chromium's Latest Patch"
date:   2024-05-19 15:22:12 +0800
categories: [RISCV]
---

The [new patch](https://chromium.googlesource.com/chromium/src/+/f68eb88e6ac1139355bad9d1f1eff784e9e82afb%5E%21/?fbclid=IwZXh0bgNhZW0CMTEAAR3oTvui6Kx-bnzP23lgZzh5Rf2Zjuayg6GG47fTOVjGqB-rbprz_355YGQ_aem_AYmswdHMwvVll9osf_FfiOD6wPfs8D7INW9uMMfQjFedPRj9-Zh2vC5lWtHZYmXbNQ5k5Si3gmexjw7Mps4R1PnP&mibextid=xfxF2i#F0) introduces several optimizations to improve the efficiency of the Adler-32 checksum calculation.


### Zero Padding to Ensure Multiple of `vl` ###

Handling remaining data can be avoided by padding zeros to make the length of the array a multiple of `vl`. The Adler-32 value depends on the accumulation of the data, so padding with zeros does not impact the result.

```c
size_t head = len & (vl - 1);
if (head > 0) {
  vuint8m2_t zero8 = __riscv_vmv_v_x_u8m2(0, vl);
  vuint8m2_t in = __riscv_vle8_v_u8m2(buf, vl);
  in = __riscv_vslideup(zero8, in, vl - head, vl);
  vuint16m4_t in16 = __riscv_vwcvtu_x(in, vl);
  a_sum = in16;
  buf += head;
}
```


### Using Two 16-Bit Accumulators Instead of 32-Bit & 16-Bit Accumulators ###

Using a combination of a 32-bit accumulator and a 16-bit accumulator in the inner loop is inefficient. Instead, two 16-bit accumulators can be used. 

**Calculating the B Part with the Trapezoidal Rule**

The accumulation of the `B` part can be calculated using the Trapezoidal rule. Assume we accumulate `x` times and the input buffer is all 255 in the worst case:
$$255 * ((1 + x) * x) / 2 <= 65535$$
and
$$255 * ((1 + 22) * 22) / 2 < 65535 < 255 * ((1 + 23) * 23) / 2$$.

Therefore, the max value of `x` will be 22.

```c
int batch = iters < 22 ? iters : 22;
// ...
while (batch-- > 0) {
  vuint8m2_t in8 = __riscv_vle8_v_u8m2(buf, vl);
  buf += vl;
  b_batch = __riscv_vadd(b_batch, a_batch, vl);
  a_batch = __riscv_vwaddu_wv(a_batch, in8, vl);
}
```


### Efficient Modulo Handling ###

The patch introduces a more efficient way to handle overflow and perform modulo operations. Instead of performing modulo after each iteration, it checks if `a_sum + a_batch` would overflow and adjusts `a_sum` accordingly.

```c
const vuint16m4_t a_overflow = __riscv_vrsub(a_sum, BASE, vl);
// ...
// Check if `a_sum + a_batch >= BASE`
vbool4_t ov = __riscv_vmsgeu(a_batch, a_overflow, vl);
a_sum = __riscv_vadd(a_sum, a_batch, vl);
// a_sum + (65536 - BASE) => a_sum - BASE in 16-bit context
a_sum = __riscv_vadd_mu(ov, a_sum, a_sum, 65536 - BASE, vl);
```


### Overflow Handling Mechanism for `b_sum` ###

`b_sum` is a 32-bit accumulator, so we don't need to prevent overflow at each iteration. Instead, we handle overflow when `b_sum` is close to exceeding $2^32-1$.

When `b_sum` >= 65535 * BASE, we have a risk to overflow. We have to prevent overflow when the inner loop is executed `b_overflow` times.

In worst case, `b_batch` would be 65535 per iterator:

$$
65535 * 22 * \text{b_overflow} < 65535 * \text{BASE} \\
=> 22 * \text{b_overflow} < \text{BASE} \\
=> \text{b_overflow} < \text{BASE}/22
$$

Therefore, the max value of `b_overflow` will be $$\text{BASE} / 23$$.

```c
const int b_overflow = BASE / 23;
int fixup = b_overflow;
// ...
b_sum = __riscv_vwaddu_wv(b_sum, b_batch, vl);
if (--fixup <= 0) {
  // not sure how this code work
  // b_sum = b_sum - BASE * (b_sum >> 16) is not mathematically eqaul to b_sum % BASE
  // it seems to serve a similar purpose in controlling calue of b_sum
  b_sum = __riscv_vnmsac(b_sum, BASE, __riscv_vsrl(b_sum, 16, vl), vl);
  fixup = b_overflow;
}
```


## Summary ##

The recent patch significantly improves the efficiency of the Adler-32 checksum calculation in Chromium, introduced by Simon Hosie from Rivos.

1. Increased Vector Register Utilization: The new implementation uses m2 instead of m1, which increases the utilization of vector registers, leading to better performance and parallelism in the inner loop.

2. Zero Padding for Vector Length Multiplicity: By padding zeros to ensure the length of the array is a multiple of the vector length (vl), the implementation ensures optimal vector processing without leftover elements that would require additional handling.

3. Efficient Overflow Prevention and Modulo Operation: The patch introduces a more efficient method to prevent overflow and handle modulo operations. The fixed inner loop executes a maximum of 22 times, making efficient use of two 16-bit accumulators.


## References ##
- [Patch landed in chromium](https://chromium.googlesource.com/chromium/src/+/f68eb88e6ac1139355bad9d1f1eff784e9e82afb%5E%21/?fbclid=IwZXh0bgNhZW0CMTEAAR3oTvui6Kx-bnzP23lgZzh5Rf2Zjuayg6GG47fTOVjGqB-rbprz_355YGQ_aem_AYmswdHMwvVll9osf_FfiOD6wPfs8D7INW9uMMfQjFedPRj9-Zh2vC5lWtHZYmXbNQ5k5Si3gmexjw7Mps4R1PnP&mibextid=xfxF2i#F0)
