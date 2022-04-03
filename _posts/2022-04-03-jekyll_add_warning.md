---
layout: post
title:  "Jekyll 增加 Warning 區塊"
author: Alex
tags: [jekyll]
---

Step 1: 
到 [https://github.com/tomjoht/documentation-theme-jekyll/tree/gh-pages/_includes](https://github.com/programminghistorian/jekyll/tree/gh-pages/_includes) 下載 `warning.html`，
放入 `_includes` 底下。

Step 2: 
將 `tomjoht/documentation-theme-jekyll/tree/gh-pages/css/bootstrap.min.css` 放到 
`assets/css` 底下。

Step 3: 
修改 `_layouts/post.html` 加入使用 `bootstrap.min.css` 來源
```html=
<link rel="stylesheet" href="/assets/css/bootstrap.min.css">
```

之後便可在文章內使用，效果如下:
{% include warning.html content="This is my warning." %}

總共有
- Note
- Tip
- Warning
- Important

等區塊可使用。

下載對應的 `html` 檔即可。

## Credit: 
- [https://idratherbewriting.com/documentation-theme-jekyll/mydoc_alerts.html#](https://idratherbewriting.com/documentation-theme-jekyll/mydoc_alerts.html#)
- [tomjoht/documentation-theme-jekyll](https://github.com/tomjoht/documentation-theme-jekyll)

