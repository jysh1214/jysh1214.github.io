---
layout: post
title:  "[CMake] Could NOT find PkgConfig (missing: PKG_CONFIG_EXECUTABLE)"
author: Alex
tags: [cmake, windows]
---

cmake-3.23.0 在 windows 底下的錯誤
```
-- Could NOT find PkgConfig (missing: PKG_CONFIG_EXECUTABLE)
```

解決辦法: 安裝 `pkg-config`

#### Step 1 ####

以系統管理員身分執行 `powershell`，
輸入
```bash=
Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
```
接著輸入
```bash=
choco install pkgconfiglite
```

之後 `pkg-config` 會安裝在
```
C:\ProgramData\chocolatey\lib\pkgconfiglite\tools\pkg-config-lite-0.28-1\bin
```

#### Step 2 ####

新增環境變數。路徑為: 編輯系統與環境變數->環境變數->系統變數->新增

新增 `PKG_CONFIG_PATH` 為 
```
C:\ProgramData\chocolatey\lib\pkgconfiglite\tools\pkg-config-lite-0.28-1\bin
```

新增 `PKG_CONFIG_EXECUTABLE` 為 
```
C:\ProgramData\chocolatey\lib\pkgconfiglite\tools\pkg-config-lite-0.28-1\bin\pkg-config.exe
```
