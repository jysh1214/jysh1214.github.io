---
layout: post
title:  "如何在 jekyll 使用 latex 數學式"
author: Alex
tags: [jekyll, latex]
---

在 jekyll 專案目錄下 `_includes/header.html` 加入
```bash=
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
```

但新的 jekyll 專案不會內建 `_includes/header.html`，所以先使用 `bundle show minima` (`minima` 為內建的風格)， 找到 `minima` 路徑:
```bash=
/Library/Ruby/Gems/2.6.0/gems/minima-2.5.1
```

將 `minima` 底下的 `_includes` 全部複製過來就可以了。

成功：
$$\nabla_\boldsymbol{x} J(\boldsymbol{x})$$

## References:

[LaTeX in Jekyll](http://www.iangoodfellow.com/blog/jekyll/markdown/tex/2016/11/07/latex-in-markdown.html)
