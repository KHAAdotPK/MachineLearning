```C++
    /*
        from_single_head_to_multi_head_attention_layer_chapter_1.md
        Q@khaa.pk
     */  
```

> **Note to Readers:** This document is a work in progress. You may encounter occasional typos and formatting inconsistencies as the content is being actively developed and refined. The focus at this stage is on technical accuracy and conceptual clarity. A thorough editorial review will be conducted in future revisions. Thank you for your understanding.

`"Readers should be aware that this article represents an ongoing project. The information and code contained herein are preliminary and will be expanded upon in future revisions."`

#### Project Overview

This series of documents provide a comprehensive analysis of a custom C++ transformer implementation, focusing on the complete pipeline from input sequence processing through encoder input/output preparation, decoder input/output preparation. The implementation represents a complete from-scratch build of the transformer architecture, including custom word embeddings, novel position encoding, and sophisticated build system architecture.

### From Single-Head to Multi-Head Attention Layer. Chapter 1.   
#### Written by, Sohail Qayum Malik
---

$$**A Hypothetical Single Head Attention Layer.**$$

#### **A Quick Clarification on Dimensions:** d<sub>k</sub>, d<sub>v</sub>, d<sub>model</sub>

Before we dive into the code, it's crucial to clear up a common point of confusion introduced by the paper's notation.

You might wonder and ask: "If we are talking about a "hypothetical single attention head", why is the paper using d<sub>k</sub> and not d<sub>model</sub> for the scaling factor?"

This is an excellent question. Let's establish some ground rules for the "Attention" function:

  1. **Query and Key:** Dimensions (d<sub>q</sub>, d<sub>k</sub>): To compute the dot product Q.K<sup>T</sup>, the "inner" dimensions must match. This means the dimension of a query vector (d<sub>q</sub>) **must be equal** to the dimension of a key vector (d<sub>k</sub>).

  2. **Value Dimension:** (d<sub>v</sub>): The dimension of the value vector d<sub>v</sub> is **independent**. It can be different from d<sub>q</sub>, d<sub>k</sub>. The output of the attention function (`softmax(...).V`) will have the dimension d<sub>v</sub>. 

  3. **The Scaling Factor:** The scaling is done by 1/&radic;d<sub>q</sub> or 1/&radic;d<sub>k</sub>. The purpose is to scale down the dot product, which is calculated using **Q** and **K**. Therefore, the scaling factor is always tied to d<sub>k</sub> (which is equal to d<sub>q</sub>).

**Connecting This to the Paper:**  

  1. **Hypothetical Single-Head Model:** If you were to build a simple model with only one attention head, you would likely set d<sub>q</sub> = d<sub>k</sub> = d<sub>v</sub> = d<sub>model</sub>. In this specific case, your scaling factor would be 1/&radic;d<sub>model</sub>.

  2. **The Paper's Approach (Preparing for Multi-Head):** The authors define "Scaled Dot-Product Attention" (3.2.1) as the general building block for their "Multi-Head Attention" (3.2.2). In the multi-head design, the model's d<sub>model</sub> is split among $h$ heads. This means each head works with smaller dimensions.

This is why the paper immediately introduces the notation:

  - d<sub>k</sub> = d<sub>model</sub> / $h$
  - d<sub>v</sub> = d<sub>value</sub> / $h$

So, when the paper says "queries and keys of dimension $d_k$", they are referring to the dimension inside one attention head. Our C++ implementation will follow this logic. We will calculate the dimensions for a single head first, and then use those dimensions for our scaling factor.  

#### 3.2 Attention

- An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

    1. The three projections **Q**, **K**, **V** from input **X** using **learned weight**s matrices

        -  **Q** = X . W<sup>Q</sup>
        -  **K** = X . W<sup>K</sup>
        -  **V** = X . W<sup>V</sup>

        #### Learned Weights?
        ```C++
        /*
         Initialization (Random Weights)
         At initialization: They're random (not learned yet)
         */
         queryWeights = Numcy::Random::randn<t>(dim); // W^Q
         keyWeights = Numcy::Random::randn<t>(dim);   // W^K 
         valueWeights = Numcy::Random::randn<t>(dim); // W^V

         // Output projection weights 
         outputWeights = Numcy::Random::randn<t>(dim); // W^O

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

        - **S** = **Q**.**K**<sup>T</sup>/&radic;d<sub>k</sub> 
        - **S** = **Q**.**K**<sup>T</sup>
        ```C++
        // Before using the last method to calulate scores from the two given, please know what you are doing.
        ```  

    3. Softmax to get weights

        - **A** = softmax(**S**)
        - The "weights" in attention (the softmax scores **A**) are computed dynamically based on the input content, not learned parameters

    4. Weighted combination or **O**utput before projection

        - **O** = **A** · **V**

    5. Output projection or Final output

        -  Y = **O** . W<sup>**O**</sup>   
        -  This final output **Y** is in the simplified equation form: output = input × weights, possibly with a bias(many frameworks include them; some simplified implementations skip them).

            -  Biases are often included in the Q, K, V projections and output projection in practice, but frequently omitted in explanations for clarity
           

    This (Y = O . W<sup>O</sup>) resembles the common neural network transformation: Y = X.W<sup>T</sup> + **b** (**bias** often omitted in attention layers), highlighting that attention can also be viewed as a series of linear layers followed by weighted combination through softmax. That is **...** the attention mechanism is fundamentally a neural network operation, but with a crucial twist

        - Standard NN: Y = X.W^T (fixed transformation)
        - Attention: Y = (softmax(Q.K^T / sqrt(d_k))·V)·W^O (dynamic, content-dependent transformation)

    That is why I previously mentioned the fact that, the "weights" in attention (the softmax scores A) are computed dynamically based on the input content, not learned parameters. This is the magic of attention!  


#### 3.2.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension d<sub>k</sub>, and values of dimension d<sub>v</sub>. We compute the dot products of the query with all keys, divide each by &radic;d<sub>k</sub>,  and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V. We compute the matrix of outputs as:

$$\text{Attention}(Q, K, V) = \text{softmax}(Q.K^T/sqrt(d_k))V$$

The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1/&radic;d<sub>k</sub>. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

While for small values of d<sub>k</sub> the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of d<sub>k</sub> [3]. We suspect that for large values of d<sub>k</sub>, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 1. To counteract this effect, we scale the dot products by 1/&radic;d<sub>k</sub>.
  
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
 cc_tokenizer::string_character_traits<char>::size_type dimensionsOfAttentionHead = dimensionsOfAttentionHead(floor((t)(d_model/num_heads))); // d_k or d<sub>k</sub>
 
 // In the following comment section d_k and d<sub>k</sub> are interchangeable. 
 /*    
    As the paper states, their algorithm is identical to standard dot-product attention except for the scaling factor of 1/sqrt(d_k). Why is this scaling necessary? 
    Imagine that the dimensions of your query and key vectors d<sub>model</sub> (in case of single head attention) and d<sub>k</sub> (in case of multi-head attention) are very large.  When you take the dot product of two random vectors in a high-dimensional space, the resulting value can become very large in magnitude. 
    
    Now, recall the next step in the attention mechanism: applying the softmax function to these scores to turn them into a probability distribution (where all attention weights sum to 1). 

    When you feed very large numbers into the softmax function, it pushes the outputs towards the extremes: 

      - One or a few outputs will get very close to 1.
      - The rest will get very close to 0.

    This is known as the "**vanishing gradient**" problem. During training, the gradients (which are used to update the model) become extremely small for the positions with attention weights near `0`, making it hard for the model to learn. The attention distribution becomes too "**sharp**" and focused, losing its ability to attend to multiple words softly.
    
    The Solution: Scaling
    ---------------------
    By dividing the dot product scores by &radic;d_k .or. &radic;d<sub>k</sub> .or. &radic;d<sub>model</sub> scale down the variance of the scores, keeping them in a range that is more stable for the `softmax` function. This results in "softer" attention distributions and much more stable gradients, which is crucial for effective training. 
  */
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
  // This comment block is only needed if you follow option 1 from the above comment block and donot not modify the scaling factor as also suggested in th above commnet block
  /**********************************************************************************************************************************************************/
  /*Note: Only query and key are scaled by the square root of the head dimension (d_k) in the forward pass                                                  */
  /*      because the attention scores are computed as the dot product of query and key.                                                                    */
  /*      This scaling prevents the dot-product values from growing too large in magnitude, which would push softmax into regions with very small gradients.*/
  /**********************************************************************************************************************************************************/ 
  // Q: X.W^Q, X is the input to the MHA layer(a.k.a ei_query)                
  query = Numcy::matmul<t>(ei_query, queryWeights);
  // K: X.W^K, X is the input to the MHA layer(a.k.a ei_key) 
  key = Numcy::matmul<t>(ei_key, keyWeights);
  // V: X.W^V, X is the input to the MHA layer(a.k.a ei_value)                
  value = Numcy::matmul<t>(ei_value, valueWeights);
```
```C++
  /*
     The attention score calculation, one of the following two.
     If you decide to go with the option 1. then please go through "MIND FART" comment block
     1. S = Q.K^T 
     2. S = Q.K^T . scaleFactor
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
         Y = O · W^O
         - O
           Output from attention before output projection (a.k.a "output")
         - W^O 
           Output projection weights (a.k.a "outputWeights")
                    
         Let Y = O.W^O // (a.k.a "Output matrix")
         In Step-1 of the backward pass, we have dL/dY = incoming_gradient when Y = OWo
   */
  output = Numcy::matmul<t>(output, outputWeights);          
```


