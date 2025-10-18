# Machine Learning.

## Neural Network Fundamentals: From Theory to Implementation

##### Project Overview.
This repository contains a series of in-depth technical articles exploring the mathematical foundations and practical implementations of neural network architectures. The content bridges the gap between theoretical understanding and code implementation, providing readers with both rigorous mathematical derivations and corresponding implementation details.

### Articles in this Series

[A Simple Neural Network](./a_simple_neural_network.md)
This introductory article demystifies **neural networks** by breaking down their fundamental structure and operations

- It explains how a basic **neural network** with one hidden layer processes information, using a concrete example with specific dimensions
- Through clear explanations of the **forward pass**, **matrix operations**, and data transformations, readers will gain a solid foundation in **neural network** concepts without requiring advanced mathematical knowledge
- Ideal for beginners and those seeking to refresh their understanding of **neural network** basics

[Fully Connected Passes.](./fully_connected_passes.txt)
A comprehensive exploration of forward and backward passes in fully connected neural networks, including:

- Detailed matrix operations with shape analysis
- Step-by-step breakdown of computational flow
- Precise mathematical formulations with practical C-style notation
- Complete derivation of gradient calculations for backpropagation

[A Line-by-Line C++ Implementation of Transformer Encoder Stack.](./transformer-encoder-implementation.md)
This article tries to bridge the gap between paper and the code:

- Shows both original (Post-LN) and modern (Pre-LN) variants
- Describes proper residual connections and layer normalization ordering

[From Single-Head to Multi-Head Attention Layer.](./from_single_head_to_multi_head_attention_layer.md)
W.I.P

 - W.I.P
 - W.I.P
 
[Fully Connected Passes in Transformer Attention Layers.](./fully_connected_passes_in_transformer_attention_layer.txt)
An advanced examination of how traditional neural network operations extend to modern transformer architectures:

- Connection between standard neural network operations and self-attention mechanisms
- Detailed breakdown of Query, Key, and Value projections
- Implementation of scaled dot-product attention with proper scaling factors
- Comprehensive backward pass derivations showing gradient flow through complex attention operations
- C++ implementation with extensive documentation and implementation notes

[Transformer Input Sequence Analysis.](https://github.com/KHAAdotPK/Transformer-Encoder-Decoder/blob/main/Implementation/ML/NLP/transformers/encoder-decoder/DOCUMENTS/input_sequence_analysis.md)
The input sequencer sets the stage for the transformer’s subsequent operations (e.g., attention mechanisms) by providing a consistent, padded, and masked inputs in numerical format. The paper emphasizes a from scratch implementation to deeply understand how pretrained embeddings are integrated into the transformer input pipeline.

[Transformer Position Encoding Analysis.](https://github.com/KHAAdotPK/Transformer-Encoder-Decoder/blob/main/Implementation/ML/NLP/transformers/encoder-decoder/DOCUMENTS/position_encoding_analysis.md)
The paper details a custom C++ position encoder as a critical component of the transformer's input processing pipeline. Transformers process tokens simultaneously (e.g., `"cat chased mouse" vs. "mouse chased cat" are same if you do not take into consideratio position encodings`) and position encoder enables the model to understand the order of words in variable-length sentences. 

[Transformer Encoder Input Analysis.](https://github.com/KHAAdotPK/Transformer-Encoder-Decoder/blob/main/Implementation/ML/NLP/transformers/encoder-decoder/DOCUMENTS/transformer_encoder_input_analysis.md)
This document analyzes debug output from a custom C++ implementation of the transformer architecture. The debug output shows the data flow through three main stages: `input sequence building`, `position encoding generation`, and final e`ncoder input preparation`. These are critical steps that enable the transformer to process sequential data effectively.

#### Transformer Decoder Stack.
1.  [Outline of constructing Decoder Input Pipeline](https://github.com/KHAAdotPK/MachineLearning/blob/main/outline_of_constructing_decoder_input_pipeline.md)
 
    - Demystifies core decoder concepts: "right-shifted inputs" for training and "look-ahead masks" for autoregressive generation
    - Provides clear visualizations of the causal attention mask structure
    - Written as a practical implementation blueprint with step-by-step examples
    - Assumes no prior knowledge, making it accessible for beginners starting from scratch
---
**Word2Vec Implementation Deep Dive: Complete Forward & Backward Propagation with C++ Code**
---
A comprehensive technical guide covering the mathematical foundations and low-level implementation of Word2Vec algorithms (Skip-gram and CBOW). Features detailed C++ code examples showing context extraction, gradient computation, weight updates, and cross-entropy loss calculation. Includes custom memory management, matrix operations, and step-by-step backpropagation with real production-level implementation details. Perfect for engineers implementing Word2Vec from scratch or students wanting to understand the algorithmic internals beyond high-level Python tutorials.

1.  [Propagation.](./propagation.md)
2.  [Cross Entropy Loss.](./cross-entropy-loss.md)
3.  [Negative Sampling in CBOW.](https://github.com/KHAAdotPK/CBOW/blob/main/DOCUMENTS/NegativeSampling.md) 	
---




