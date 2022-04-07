---
layout: post
title:  "cuda-sample code 跟本地 cuda 版本不匹配"
author: Alex
tags: [cuda]
---

從 [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) 下載 sample code，
可以從 `git tags` 選擇對應的 `cuda` 版本。

<center><img src="/assets/images/2022-04-07-cuda-sample-version/cuda-sample-tags.png" width="450"></center>

那想用舊的 `cuda` 版本執行較新的 sample 呢? 例如電腦安裝 11.1 版本但想跑 11.5 版本的 sample code。

解決辦法:

1. 安裝相對應的 `cuda` 版本，這裡不詳細討論
2. 修改目錄底下的 `vcxproj` 檔 -> 將所有的 `CUDA 11.5.props` 改成 `CUDA 11.1.props` 即可

改完成後編譯，可能會有一些編譯選項不支援，至 `vcxproj` 檔將對應的 
```
<AdditionalOptions>–<FLAG> <VALUE></AdditionalOptions>
``` 
拿掉即可。


