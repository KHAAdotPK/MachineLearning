```C++
    /*
        from_single_head_to_multi_head_attention_layer.md
        Q@khaa.pk
     */  
```

`"Readers should be aware that this article represents an ongoing project. The information and code contained herein are preliminary and will be expanded upon in future revisions."`

#### Project Overview

This series of documents provide a comprehensive analysis of a custom C++ transformer implementation, focusing on the complete pipeline from input sequence processing through encoder input/output preparation, decoder input/output preparation. The implementation represents a complete from-scratch build of the transformer architecture, including custom word embeddings, novel position encoding, and sophisticated build system architecture.


### From Single-Head to Multi-Head Attention Layer
#### Written by, Sohail qayum malik
---

#### Attention (from the paper)

- An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

    1. The three projections **Q**, **K**, **V** from input **X** using **learned weight**s matrices

        -  **Q** = X . W^Q 
        -  **K** = X . W^K 
        -  **V** = X . W^V

        #### Learned Weights?
        ```C++
        /*
         Initialization (Random Weights)
         At initialization: They're random (not learned yet)
         */
         queryWeights = Numcy::Random::randn<t>(dim);
         keyWeights = Numcy::Random::randn<t>(dim);
         valueWeights = Numcy::Random::randn<t>(dim);

         outputWeights = Numcy::Random::randn<t>(dim);

        /*
         The weights learn through this process of gradient descent! 
         During training: They're learning (being optimized)
         After training: They're learned (optimized)
         */

         //  Backpropagation (Learning)? Compute how wrong we were in our predictions
         //  ------------------------------------------------------------------------
         /*`                                                       
          Learning Rate Scaling:
          - During weight updates, we multiply the computed gradients by the learning rate.
          - This controls the size of the update step taken towards minimizing the loss.
          - A smaller learning rate means smaller updates (more stable but slower learning).
          - A larger learning rate means bigger updates (faster but can cause instability if too large).
         - Mathematically:  new_weight = old_weight - learning_rate * gradient
          */
         gradient_query_weights = gradient_query_weights * learning_rate;
         this->queryWeights = this->queryWeights - gradient_query_weights;

         gradient_key_weights = gradient_key_weights * learning_rate;
         this->keyWeights = this->keyWeights - gradient_key_weights;

         gradient_value_weights = gradient_value_weights * learning_rate;
         this->valueWeights = this->valueWeights - gradient_value_weights;
    ```        

    2. The attention score calculation

        - **S** = **Q**.**K**^T/sqrt(d_k) 

    3. Softmax to get weights

        - **A** = softmax(**S**)
        - The "weights" in attention (the softmax scores **A**) are computed dynamically based on the input content, not learned parameters

    4. Weighted combination or **O**utput before projection

        - **O** = **A** · **V**

    5. Output projection or Final output

        -  Y = **O** . W^**O**   
        -  This final output **Y** is in the simplified equation form: output = input × weights, possibly with a bias(many frameworks include them; some simplified implementations skip them).

            -  Biases are often included in the Q, K, V projections and output projection in practice, but frequently omitted in explanations for clarity
           

    This (Y = O . W^O) resembles the common neural network transformation: Y = XW^T + b (bias often omitted in attention layers), highlighting that attention can also be viewed as a series of linear layers followed by weighted combination through softmax. That is**...** the attention mechanism is fundamentally a neural network operation, but with a crucial twist

        - Standard NN: Y = XW (fixed transformation)
        - Attention: Y = (softmax(QK^T / sqrt(d_k)) · V) · W^O (dynamic, content-dependent transformation)

    That is why I previously mentined the fact that, the "weights" in attention (the softmax scores A) are computed dynamically based on the input content, not learned parameters. This is the magic of attention!         

