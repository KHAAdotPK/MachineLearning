# From Single-Head to Multi-Head Attention Layer. Chapter 2
#### Written by Sohail Qayum Malik

> **Note to Readers:** This document is a work in progress, part of an ongoing series on a custom C++ transformer implementation. It extends the concepts introduced in Chapter 1, focusing on multi-head attention. Expect minor typos or formatting issues, which will be refined in future revisions. Thank you for your patience.

`"Readers should be aware that this article represents an ongoing project. The information and code contained herein are preliminary and will be expanded upon in future revisions."`

### Extending to Multi-Head Attention

#### Update on Chapter 2: Why It’s Not Available Yet

Hey readers, Chapter 2 on multi-head attention isn’t ready just yet ~~, and here’s why: I’m working through some tricky details on reshaping the encoder inputs for the multi-head attention layer. My input is a row vector of shape `(3, 16)` (3 tokens, 16 features), and splitting it into multiple heads (like 8 or 6) requires careful handling to ensure the feature dimension divides evenly. For example, with 8 heads, each head gets a clean `(3, 2)` slice, but 6 heads causes issues since `16 / 6` isn’t an integer. I’m refining the logic to reshape the input properly (possibly with padding for cases like 6 heads) to make the code robust and clear. This is part of my larger C++ Transformer project, and I want to get it right before sharing. Stay tuned for the update, and thanks for following along!~~

#### 3.2.2 Multi-Head Attention

Instead of performing a single attention function with d<sub>model</sub>-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to d<sub>k</sub>, d<sub>k</sub> and d<sub>v</sub> dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_v$-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2. 

Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

$$\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Where the projections are parameter matrices W<sub>i</sub><sup>Q</sup> **&#8712;** $R$<sup>d<sub>model</sub>**x**d<sub>k</sub></sup>, W<sub>i</sub><sup>K</sup> **&#8712;** $R$<sup>d<sub>model</sub>**x**d<sub>k</sub></sup>, W<sub>i</sub><sup>V</sup> **&#8712;** $R$<sup>d<sub>model</sub>**x**d<sub>v</sub></sup> ($please$ $refer$ $to$ **( 1C )**) and W<sup>O</sup> **&#8712;** $R$<sup>hd<sub>v</sub>**x**d<sub>model</sup>   
In this work we employ $h = 8$ parallel attention heads. For each of these we use $d_k = d_v = d_{\text{model}}/h = 64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

**Standard Q, K, V dimensions:** In the standard multi-head attention mechanism (e.g., as defined in the Transformer model from "Attention is All You Need"), the input matrices ( Q ), ( K ), and ( V ) typically have the same second dimension, d<sub>model</sub> meaning:

- Q, K, V &#8712; $R$<sup>seq_len**x**d<sub>model</sub></sup> 
- The projection matrices are then:

  - $W$<sub>i</sub><sup>Q</sup>, $W$<sub>i</sub><sup>K</sup> &#8712; $R$<sup>d<sub>model</sub>**x**d<sub>k</sub></sup> 
  - $W$<sub>i</sub><sup>V</sup> &#8712; $R$<sup>d<sub>model</sub>**x**d<sub>v</sub></sup>

This ensures that the rows of all projection matrices correspond to d<sub>model</sub>, the ~~word~~ embedding dimension of the inputs. 

**V can have diffrent length of ~~word~~ embeddings:** In the examples of this article $V$ has different second dimensions. The attention mechanism doesn’t strictly require ( V ) to have the same number of columns as ( Q ) and ( K ), because: 

- The attention scores are computed using QW<sub>i</sub><sup>Q</sup> and KW<sub>i</sub><sup>K</sup> (For both the second dimensions need to be same beuase we take product of QW<sub>i</sub><sup>Q</sup> and $transpose$ of KW<sub>i</sub><sup>K</sup>).
- VW<sub>i</sub><sup>V</sup> only needs to be compatible with the $softmax$ output: $softmax$(QW<sub>i</sub><sup>Q</sup>, **transpose**(KW<sub>i</sub><sup>K</sup>)).**VW<sub>i</sub><sup>V</sup>**.

This flexibility allows ( V ) to have a different embedding dimension d<sub>v</sub>**x**$h$

### Multi-Head Attention Mechanics

The multi-head attention mechanism is defined as follows:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$

where each head is computed as:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

The projections are parameter matrices with the following dimensions:
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ 
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ **( 1C )**: This notation can be confusing. The first dimension, d<sub>model</sub> must match the feature dimension of the input $V$ matrix. This is inconsistent with the standard, where $V$ typically shares the same d<sub>model</sub> as inputs $Q$ and $K$. As shown in the example, $V$ **&#8712;** $R$<sup>3**x**8</sup> while $Q$, $K$ **&#8712;** $R$<sup>3**x**16</sup>. Therefore, $W$<sub>i</sub><sup>V</sup> should be **&#8712;** $R$<sup>8**x**d<sub>v</sub></sup>. In this specific case, the input dimension $8$ corresponds to $h$.d<sub>v</sub>, so a more precise notation for this example would be $W$<sub>i</sub><sup>V</sup> **&#8712;** $R$<sup>hd<sub>v</sub>**x**d<sub>v</sub></sup>.
- $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$

## Example Calculation

Suppose we have the following input dimensions:
- Query matrix: $Q \in \mathbb{R}^{3 \times 16}$
- Key matrix: $K \in \mathbb{R}^{3 \times 16}$
- Value matrix: $V \in \mathbb{R}^{3 \times 8}$
- Number of heads: $h = 8$
- Dimensions per head:
  - $d_q = d_k = 2$
  - $d_v = 1$

### Projection Matrices
The projection matrices for the first head ($i=1$) are:
- $W_1^Q \in \mathbb{R}^{16 \times 2}$
- $W_1^K \in \mathbb{R}^{16 \times 2}$
- $W_1^V \in \mathbb{R}^{8 \times 1}$ please refer to **( 1C )**

~~(Note: The dimensions of $W^Q$, $W^K$, and $W^V$ in the original text were given as $16 \times 16$, $16 \times 16$, and $8 \times 8$, respectively, but these seem inconsistent with the head-specific projections. We use the head-specific dimensions for calculations.)~~. $For$ $discussion$, $please$ $refer$ $to$ **( 2C )**

### Head Computation
For the first head ($i=1$):
- Compute $QW_1^Q$:
  $$QW_1^Q = (3 \times 16) \cdot (16 \times 2) = 3 \times 2$$
- Compute $KW_1^K$:
  $$KW_1^K = (3 \times 16) \cdot (16 \times 2) = 3 \times 2$$
- Compute $VW_1^V$:
  $$VW_1^V = (3 \times 8) \cdot (8 \times 1) = 3 \times 1$$

The attention mechanism for the first head is:

$$
\text{head}_1 = \text{Attention}(QW_1^Q, KW_1^K, VW_1^V) = \text{softmax}\left(\frac{(QW_1^Q)(KW_1^K)^T}{\sqrt{d_k}}\right)(VW_1^V)
$$

- Compute $(QW_1^Q)(KW_1^K)^T$: $$(3 \times 2) \cdot (2 \times 3) = 3 \times 3$$
- Apply softmax to the scaled result (divided by $\sqrt{d_k} = \sqrt{2}$), then multiply by $VW_1^V$: $$(3 \times 3) \cdot (3 \times 1) = 3 \times 1$$

Thus, $\text{head}_1 \in \mathbb{R}^{3 \times 1}$.

### Concatenation Across Heads
Since there are $h = 8$ heads, and each head outputs a matrix of shape $3 \times 1$, the concatenation of all heads is:

$$
\text{Concat}(\text{head}_1, \dots, \text{head}_8) \in \mathbb{R}^{3 \times 8}
$$

### Final Output
The output projection matrix is:

$$
W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}} = \mathbb{R}^{8 \cdot 1 \times 16} = \mathbb{R}^{8 \times 16}
$$

The final output of the multi-head attention is:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_8)W^O = (3 \times 8) \cdot (8 \times 16) = 3 \times 16
$$

( 2C ) **Alternative Interpretation of Weight Dimensions** 

While the standard formulation uses separate projection matrices for each head, one can envision the weight matrices $W^Q$, $W^K$, and $W^V$ as consolidated repositories containing all head projections. In this view:

- $W^Q \in \mathbb{R}^{16 \times 16}$ can be thought of as containing all 8 head projections ($W_i^Q \in \mathbb{R}^{16 \times 2}$) concatenated along the second dimension
- Similarly, $W^K \in \mathbb{R}^{16 \times 16}$ consolidates all $W_i^K$ projections
- $W^V \in \mathbb{R}^{8 \times 8}$ contains all $W_i^V$ projections

During computation, the specific slices corresponding to each head's dimensions ($d_q=2$, $d_k=2$, $d_v=1$) are extracted from these master weight matrices. This perspective is particularly relevant when considering:

1. **Hypothetical Single-Head Self-Attention**: For a single head with full dimensionality, these dimensions become directly applicable
2. **Weight Organization**: The matrices serve as centralized repositories from which head-specific projections are drawn according to the required $d_q$, $d_k$, and $d_v$ dimensions
3. **Implementation Flexibility**: This approach provides a unified storage mechanism that can be partitioned dynamically based on the head configuration

Though this differs from the standard formulation where each head has explicitly separate learned projections, it offers an alternative conceptual framework for understanding the weight organization in multi-head attention systems.

---

```C++
```
```C++
```