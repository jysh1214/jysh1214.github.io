---
layout: post
title:  "利用 macro 實現跨編譯器兼容"
author: Alex
tags: [c++]
---

開發 C++ 程式碼時，很有可能會因為不同編譯器所支援的 `feature` 不同，
而讓程式碼不能在不同編譯器上相容。

利用 `Language features macro` 可以判斷目前的編譯器是否支援該項 `feature`，
而決定是否啟用程式碼:
```c++=
#if __cpp_constexpr // 若編譯器支援 constexpr 才啟用下段程式碼
if constexpr (condition) {
  // ...
}
#endif
```

#### Refereces ####
- [C++ compiler support](https://en.cppreference.com/w/cpp/compiler_support)
- [Language features](https://en.cppreference.com/w/cpp/feature_test)
