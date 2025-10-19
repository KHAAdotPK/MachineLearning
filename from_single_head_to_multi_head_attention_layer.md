```C++
    /*
        from_single_head_to_multi_head_attention_layer.md
        Q@khaa.pk
     */  
```

> **Note to Readers:** This document is a work in progress. You may encounter occasional typos and formatting inconsistencies as the content is being actively developed and refined. The focus at this stage is on technical accuracy and conceptual clarity. A thorough editorial review will be conducted in future revisions. Thank you for your understanding.

`"Readers should be aware that this article represents an ongoing project. The information and code contained herein are preliminary and will be expanded upon in future revisions."`

#### Project Overview

This series of documents provide a comprehensive analysis of a custom C++ transformer implementation, focusing on the complete pipeline from input sequence processing through encoder input/output preparation, decoder input/output preparation. The implementation represents a complete from-scratch build of the transformer architecture, including custom word embeddings, novel position encoding, and sophisticated build system architecture.

### From Single-Head to Multi-Head Attention Layer
#### Written by, Sohail Qayum Malik
---

#### 3.2 Attention

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
    2. The attention score calculation, one of the following two

        - **S** = **Q**.**K**^T/sqrt(d_k) 
        - **S** = **Q**.**K**^T 
        ```C++
        // Before using the last method to calulate scores from the two given, please know what you are doing.
        ```  

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

    That is why I previously mentioned the fact that, the "weights" in attention (the softmax scores A) are computed dynamically based on the input content, not learned parameters. This is the magic of attention!  


#### 3.2.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension d<sub>k</sub>, and values of dimension d<sub>v</sub>. We compute the dot products of the query with all keys, divide each by sqrt(`d_k`), and apply a softmax function to obtain the weights on the values.

```C++
  /*	
     @d_model, name from the paper "Attention is all you need" we call it "dimensionsOfTheModel".	
     d_model is a Hyperparameter:
      - Represents the dimensionality of the model's embeddings.
      - These are not pre-trained like Word2Vec they start as random numbers and are trained from scratch...
        typically small numbers from a Gaussian or uniform distribution).
      - Common values: 128, 256, 512, 1024 (depends on model size)
      - Higher d_model = more capacity, but also more computation
      - During training, these vectors are updated via backpropagation to capture meaningful semantic relationships.	
   */ 
 /*
    Head Dimension Type
    -------------------
    // The statement following this comment block seems unnecessarily complex and could be simplified to...
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfAttentionHead = d_model / num_heads;
    // Add assertion to ensure even division
    assert(d_model % num_heads == 0);
  */
 cc_tokenizer::string_character_traits<char>::size_type dimensionsOfAttentionHead = dimensionsOfAttentionHead(floor((t)(d_model/num_heads)));
 t scaleFactor = (1.0 / std::sqrt(dimensionsOfAttentionHead));	
```
```C++
   DIMENSIONS dim = DIMENSIONS{d_model, d_model, NULL, NULL}; 
   
   // Projection matrices for Q, K, V (respectively W^Q, W^K, W^V) 
   Collective<t> queryWeights, keyWeights, valueWeights, outputWeights;

   queryWeights = Numcy::Random::randn<t>(dim);
   keyWeights = Numcy::Random::randn<t>(dim);
   valueWeights = Numcy::Random::randn<t>(dim);
  
   outputWeights = Numcy::Random::randn<t>(dim);

  /*

                                            -: CONSIDER THIS COMMENT BLOCK MY MIND FART BEGINS HERE :-

    Use one and only one of the following scaling strategies:

    1. Option 1: Scale Q and K during projections:
       - Q = X * W^Q * scaleFactor;
       - K = X * W^K * scaleFactor;
       - V = X * W^V (no scaling needed in either case)
       scores = query · key^T;
                    
    2. Option 2: Scale scores after computing them (From "Attention is all you need") : 
       - Q = X * W^Q
       - K = X * W^K
       - V = X * W^V (no scaling needed in either case)
       scores = query · key^T * scaleFactor;  
    
    Please Note:- if you follow Option 1, then your scaling factor would be...
    - t scaleFactor = 1.0 / std::sqrt(std::sqrt(static_cast<t>(dimensionsOfAttentionHead)));
    // Or more clearly:
    - t scaleFactorForQK = 1.0 / std::pow(static_cast<t>(dimensionsOfAttentionHead), 0.25);

    And that is because when calculating scores the dot product of Q and K would turn 1/sqrt(d_k) into 1/d_k (sqrt(d_k)^2 = d_k). So it is better to stay with option 2, which is also inline with the actual paer as well.
    
                                            -: CONSIDER THIS COMMENT BLOCK MY MIND FART ENDS HERE :-
   */
  // This comment block is only needed if you follow option 1 from the above comment block and donot not modify the scalling factor as also suggested in th above commnet block
  /**********************************************************************************************************************************************************/
  /*Note: Only query and key are scaled by the square root of the head dimension (d_k) in the forward pass                                                  */
  /*      because the attention scores are computed as the dot product of query and key.                                                                    */
  /*      This scaling prevents the dot-product values from growing too large in magnitude, which would push softmax into regions with very small gradients.*/
  /**********************************************************************************************************************************************************/ 
  // Q: XW^Q, X is the input to the MHA layer(a.k.a ei_query)                
  query = Numcy::matmul<t>(ei_query, queryWeights);
  // K: XW^K, X is the input to the MHA layer(a.k.a ei_key) 
  key = Numcy::matmul<t>(ei_key, keyWeights);
  // V: XW^V, X is the input to the MHA layer(a.k.a ei_value)                
  value = Numcy::matmul<t>(ei_value, valueWeights);
```
```C++
  /*
     The attention score calculation, one of the following two...
     - S = Q.K^T 
     - S = Q.K^T . scaleFactor
   */
  /* Compute scaled dot-product attention scores */
  scores = Numcy::matmul<t>(query, Numcy::transpose(key)) * scaleFactor; 
```
```C++
  /*
       Softmax to get weights
       - A = softmax(S)      
      Attention weights, which are the normalized scores indicating how much focus each word should receive.
      These weights are sometimes called just "attention weights"  and other times are called "cached attention weights"
   */    
  // Apply softmax to get (attention weights a.k.a "A")  
  attention_weights = softmax<t>(scores);
  /*
    - O = A · V                              
      Output from attention before output projection
   */
  output = Numcy::matmul<t>(attention_weights, value); 
```
```C++
  /*
       Final Projected Output: Attention Projection Output = O*Wo = OWo Matrix
         Y = O · Wo
         - O
           Output from attention before output projection (a.k.a "output")
         - Wo 
           Output projection weights (a.k.a "outputWeights")
                    
         Let Y = O*Wo = OWo Matrix (a.k.a "Output matrix")
         In Step-1 of the backward pass, we have dL/dY = incoming_gradient when Y = OWo
   */
  output = Numcy::matmul<t>(output, outputWeights);          
```


