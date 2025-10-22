# From Single-Head to Multi-Head Attention Layer. Chapter 2
#### Written by Sohail Qayum Malik

> **Note to Readers:** This document is a work in progress, part of an ongoing series on a custom C++ transformer implementation. It extends the concepts introduced in Chapter 1, focusing on multi-head attention. Expect minor typos or formatting issues, which will be refined in future revisions. Thank you for your patience.

`"Readers should be aware that this article represents an ongoing project. The information and code contained herein are preliminary and will be expanded upon in future revisions."`

### Extending to Multi-Head Attention

#### Update on Chapter 2: Why It’s Not Available Yet

Hey readers, Chapter 2 on multi-head attention isn’t ready just yet, and here’s why: I’m working through some tricky details on reshaping the encoder inputs for the multi-head attention layer. My input is a row vector of shape `(3, 16)` (3 tokens, 16 features), and splitting it into multiple heads (like 8 or 6) requires careful handling to ensure the feature dimension divides evenly. For example, with 8 heads, each head gets a clean `(3, 2)` slice, but 6 heads causes issues since `16 / 6` isn’t an integer. I’m refining the logic to reshape the input properly (possibly with padding for cases like 6 heads) to make the code robust and clear. This is part of my larger C++ Transformer project, and I want to get it right before sharing. Stay tuned for the update, and thanks for following along!


