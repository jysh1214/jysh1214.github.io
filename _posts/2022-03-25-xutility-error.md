---
layout: post
title:  "xutility ERROR"
author: Alex
tags: [cmake]
---

在 `Visual Studio 2019` 中編譯 `CUDA` 產生以下錯誤:
- `expected a "("`
- `identify "_Verify_range" is undefined`

錯誤來源在 `xutility` 之內。

<center><img src="/assets/images/2022-03-25-xutility-error/xutility-error.png" width="450"></center>

雙擊錯誤查看 `xutility` 內容可發現，第 1306 行:
```c++=
template <class _Iter, class _Sentinel>
constexpr void _Adl_verify_range(const _Iter& _First, const _Sentinel& _Last) {
    // check that [_First, _Last) forms an iterator range
    if constexpr (_Range_verifiable_v<_Iter, _Sentinel>) {
        _Verify_range(_First, _Last);
    }
}
```

因為 `if constexpr` 是 `C++17` 才有的 `feature`，故在 `Camke` 內指定
```cmake=
set(CMAKE_CUDA_STANDARD 17)
```
即可。