---
layout: post
title:  "在 Synology NAS 上建立 Git 專案"
author: Alex
tags: [git]
---

1. 到 `套件中心` 下載 `git server` 套件
2. 開啟 `ssh` 功能
3. 新增一個目錄給 `git` 使用
4. 在 `NAS` 新增一個專案並 `clone` 到 `host`
    ```
    ssh <YOUR_ACCOUNT>@<IP>
    cd /volume1/<YOUR_GIT_DIR>/<YOUR_REPOSITORY>
    git --bare init
    git clone <YOUR_ACCOUNT>@<IP>:/volume1/<YOUR_GIT_DIR>/<YOUR_REPOSITORY>
    ```