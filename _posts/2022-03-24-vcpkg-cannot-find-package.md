---
layout: post
title:  "vcpkg 安裝套件後找不到"
author: Alex
tags: [vcpkg, cmake]
---

使用 `vcpkg` 安裝套件後，如果在 `CMake` 內找不到套件，可能為以下幾種原因: 

#### 原因 1. 系統找不到對應的 `CMake` 檔 ####

解決辦法: `CMake` 添加指令 
```bash=
-D CMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```
讓 `CMake` 搜尋 `vcpkg` 安裝套件。

#### 原因 2. `CMake` 指定為 `x64` 專案，但 `vcpkg` 安裝為 `x86` 版本 ####

解決辦法: 安裝指定 `x64` 版本套件。
```bash=
vcpkg install XXX:x64-windows
```