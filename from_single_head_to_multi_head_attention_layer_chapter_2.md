# From Single-Head to Multi-Head Attention Layer. Chapter 2
#### Written by Sohail Qayum Malik

> **Note to Readers:** This document is a work in progress, part of an ongoing series on a custom C++ transformer implementation. It extends the concepts introduced in Chapter 1, focusing on multi-head attention. Expect minor typos or formatting issues, which will be refined in future revisions. Thank you for your patience.

`"Readers should be aware that this article represents an ongoing project. The information and code contained herein are preliminary and will be expanded upon in future revisions."`

### Extending to Multi-Head Attention

#### Update on Chapter 2: Why It’s Not Available Yet

Hey readers, Chapter 2 on multi-head attention isn’t ready just yet, and here’s why: I’m working through some tricky details on reshaping the encoder inputs for the multi-head attention layer. My input is a row vector of shape `(3, 16)` (3 tokens, 16 features), and splitting it into multiple heads (like 8 or 6) requires careful handling to ensure the feature dimension divides evenly. For example, with 8 heads, each head gets a clean `(3, 2)` slice, but 6 heads causes issues since `16 / 6` isn’t an integer. I’m refining the logic to reshape the input properly (possibly with padding for cases like 6 heads) to make the code robust and clear. This is part of my larger C++ Transformer project, and I want to get it right before sharing. Stay tuned for the update, and thanks for following along!

#### 3.2.2 

Instead of performing a single attention function with d<sub>model</sub>-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to d<sub>k</sub>, d<sub>k</sub> and d<sub>v</sub> dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_v$-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2. 

Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

$$\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Where the projections are parameter matrices W<sub>i</sub><sup>Q</sup> **&#8712;** $R$<sup>d<sub>model</sub>**x**d<sub>k</sub></sup>, W<sub>i</sub><sup>K</sup> **&#8712;** $R$<sup>d<sub>model</sub>**x**d<sub>k</sub></sup>, W<sub>i</sub><sup>V</sup> **&#8712;** $R$<sup>d<sub>model</sub>**x**d<sub>v</sub></sup> and W<sup>O</sup> **&#8712;** $R$<sup>hd<sub>v</sub>**x**d<sub>model</sup>   
In this work we employ $h = 8$ parallel attention heads. For each of these we use $d_k = d_v = d_{\text{model}}/h = 64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.



```C++
```
```C++
```


