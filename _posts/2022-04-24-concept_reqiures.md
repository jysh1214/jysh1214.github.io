---
layout: post
title:  "C++ 20 concept & requires"
author: Alex
tags: [c++]
---

在 c++ 20 使用 `tamplate` 時，可以透過 `concept`、`requires` `keyword` 
在編譯時期對 `tamplate` 加以規範。

例如需要對某個 `class` 開發一個排序函數，定義為:
```c++=
/**
 * @tparam _Elem the type of data
 * @tparam _Container the type of container
 * @tparam _Compare compare function
 */
template <typename _Elem,
    typename _Container,
    typename _Comp>
void mySort(_Container& container, _Comp comp);
```

為了確保 `_Container` 內的資料型態與 `_Elem` 一樣可以為其加上 `concept`:
```c++=
template <typename _Elem, typename _Container>
concept IsSameDataType = std::is_same_v<_Elem, typename _Container::value_type>;
```

為 `mySort` 加上 `concept`:
```c++=
template <typename _Elem,
    typename _Container,
    typename _Comp>
requires IsSameDataType<_Elem, _Container>
void mySort(_Container& container, _Comp comp) {
    // ...
}
```

這樣一來便可以在編譯時期判斷資料型態是否一致:
```c++=
std::vector<MyClass> vec{ /*...*/ };
auto comp = [](MyClass& a, MyClass& b){
    // ...
};

mySort<MyClass, std::vector<MyClass>, decltype(comp)>(vec, comp); // OK
mySort<MyClass, std::vector<int>, decltype(comp)>(vec, comp); // error: 資料型態不一致
```

#### Refereces ####
- [Advanced Template Use](http://www.icce.rug.nl/documents/cplusplus/cplusplus23.html)
- [Templates](https://timsong-cpp.github.io/cppwp/temp.inst#17)