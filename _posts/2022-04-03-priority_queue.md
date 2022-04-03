---
layout: post
title:  "std::priority_queue 用法"
author: Alex
tags: [cpp]
---

`std::priority_queue` 定義為
```
priority_queue<Type, Container, Compare>
```
- `Type`: 數據類型
- `Container`: 實現用的容器，一般使用 `vector`，默認為 `vector`
- `Compare`: 比較用的函數，默認為 `max heap`

底層是由 `heap` 所實現的。
以下為常用的幾種使用方法:

#### 一般使用 ####
```c++=
std::vecotr<int> v{ ... };
std::priority_queue<int> max_heap(v.begin(), v.end());
```

上方程式碼等於
```c++=
#include <functional>

std::vecotr<int> v{ ... };
std::priority_queue<int, std::vector<int>, std::less<int>> max_heap;
for (int elem : v) max_heap.push(elem); 
```

{% include important.html content="要使用 `std::less` 需要 `#include <functional>`。" %}

要實現 `min heap` 則需要修改 `Compare`:
```c++=
std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap;
```

範例 [Leetcode 2208](https://leetcode.com/problems/minimum-operations-to-halve-array-sum/):
```c++=
class Solution {
public:
    int halveArray(vector<int>& nums) {
        int count = 0;
        priority_queue<double> q(nums.begin(), nums.end());
        double sum = 0;
        sum = accumulate(nums.begin(), nums.end(), sum);
        double sumHalf = sum / 2;
        while (sumHalf > 0) {
            double half = q.top() / 2;
            q.pop();
            if (half > 0) q.push(half);
            sumHalf -= half;
            ++count;
        }
        
        return count;
    }
};
```

{% include note.html content="
這裡需要注意的是，`max_heap` 要使用 `std::less` 而 `min heap` 要使用 `std::greater`。
如果熟悉使用 `lambda` 函數做排序的話，可以知道使用 `std::less` 排序為升冪，
`std::greater` 為降冪，而 `std::priority_queue` 本質為 `queue`(只是內有優先順序)，
尾端為 top，跟 `sort` 排序出來的 `layout` 是一樣的。
" %}


#### 使用自定義的排序 ####
例如我們設計一個 `linked list`
```c++=
struct ListNode {
    ListNode* next;
    int value;
};
```

我們希望使用 `ListNode` 的 `value` 去實現一個 `min heap`，
這時就需要自己設計比較函數。

```c++=
auto comp = [](ListNode* a, ListNode* b){
    return a->val > b->val;
};

std::priority_queue<ListNode*, vector<ListNode*>, decltype(comp)> q(comp);
```

範例 [Leetcode 23](https://leetcode.com/problems/merge-k-sorted-lists/):
```c++=
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        auto comp = [](ListNode* a, ListNode* b){
            return a->val > b->val;
        };
        
        priority_queue<ListNode*, vector<ListNode*>, decltype(comp)> q(comp);
        for (ListNode* node : lists) {
            if (node) q.push(node);
        }
        
        ListNode* node = new ListNode(0);
        ListNode* head = node;
        while (!q.empty()) {
            node->next = q.top();
            q.pop();
            node = node->next;
            if (node->next) {
                q.push(node->next);
            }
        }
        
        return head->next;
    }
};
```

{% include note.html content="
如果使用 `std::pair` ，那默認比較函數是使用 `first` 的數值做比較。
" %}