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

## Multi-Head Attention Mechanics

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
EncoderLayer(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads, t dropout_rate)
                : attention(d_model, num_heads),      /* Initialize attention module */
                  ffn(d_model, dropout_rate),         /* Initialize FeedForward Network */
                  /*norm1*/ attention_norm(d_model),  /* Initialize Layer Normalization 1 */
                  /*norm2*/ ffn_norm(d_model),        /* Initialize Layer Normalization 2 */
                  dimensionsOfTheModel(d_model), 
                  numberOfAttentionHeads(num_heads), 
                  dropOutRate(dropout_rate),
                  multiHeadAttentionListHead(NULL) {

    MultiHeadAttentionList<t>* current = NULL;

    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < numberOfAttentionHeads; i++)
    {
        if (current == NULL)
        {                    
            current = reinterpret_cast<MultiHeadAttentionList<t>*>(cc_tokenizer::allocator<char>().allocate(sizeof(MultiHeadAttentionList<t>)));
            multiHeadAttentionListHead = current;
            current->previous = NULL;                    
        }
        else
        {                 
            current->next = reinterpret_cast<MultiHeadAttentionList<t>*>(cc_tokenizer::allocator<char>().allocate(sizeof(MultiHeadAttentionList<t>)));
            current->next->previous = current;
            current = current->next;
        }
                
        current->next = NULL;
        /*
            In a Transformer-based encoder (such as in BERT or GPT-like models), each encoder layer consists of multiple sublayers, typically:
            1. Self-Attention Layer, it consits of multiple of attention heads
            2. Feedforward Layer
            3. Layer Normalization (before or after these)
         */    
                current->ptr = new Attention<t>(dimensionsOfTheModel, numberOfAttentionHeads);     
    }
}
```
```C++
Collective<t> forward(Collective<t>& ei/*, Collective<t>& mask*/, Collective<t>& attentionMaskInputSequence, ENCODER_LAYER_NORM_POSITION_TYPE norm_position = PreAttentionAndFeedForwardNetwork, bool is_training = true) throw (ala_exception)
{   
    /*                
     * Each head needs to get a clean slice of the feature dimension. If the number of heads doesn't divide evenly into the feature size, then without padding or adjustments, some heads would end up with fractional features, which isn't valid.
     * Adding padding or some adjustment/resolution ensures that each head gets equal numbers of features, thus maintaining the integrity of the multi-head attention mechanism.
     * At the moment, an exception is just being thrown if the number of heads does not divide evenly into the feature size
     */           
    if (ei.getShape().getNumberOfColumns() % numberOfAttentionHeads)
    {
        throw ala_exception(cc_tokenizer::String<char>("EncoderLayer<t>::forward(Collective<t>&, Collective<t>&, ENCODER_LAYER_NORM_POSITION_TYPE, bool) Error: The number of columns \"") + cc_tokenizer::String<char>(ei.getShape().getNumberOfColumns()) + cc_tokenizer::String<char>("\" must be evenly divisible by the number of attention heads \"") + cc_tokenizer::String<char>(numberOfAttentionHeads) + cc_tokenizer::String<char>("\" for multi-head attention."));
    }

    /*
     * Ensure the input feature dimension matches the model's expected dimension
     */
    if (ei.getShape().getNumberOfColumns() != dimensionsOfTheModel)
    {
        throw ala_exception(cc_tokenizer::String<char>("EncoderLayer<t>::forward(Collective<t>&, Collective<t>&, ENCODER_LAYER_NORM_POSITION_TYPE, bool) Error: The number of input columns  \"") + cc_tokenizer::String<char>(ei.getShape().getNumberOfColumns()) + cc_tokenizer::String<char>("\" must be equal to the model dimension \"") + cc_tokenizer::String<char>(dimensionsOfTheModel) + cc_tokenizer::String<char>("\"."));
    }

    // Pointer to traverse the linked list of attention heads
    MultiHeadAttentionList<t>* current = multiHeadAttentionListHead;

    // Variables to manage array dimensions and slicing
    DIMENSIONSOFARRAY dimensionOfArray; 
    DIMENSIONS /*dimension_ei_slice,*/ dimension_qkv_weights, dimension_qkv_slice;

    // Collective objects for storing concatenated results and individual slices
    Collective<t> ei_concatenated/*, ei_slice*/, q_slice, k_slice, v_slice;
    // Counter for tracking slice positions
    cc_tokenizer::string_character_traits<char>::size_type i = 0;

    // Projection matrices for Q, K, V (respectively W^Q, W^K, W^V) and "output projection weights" matrix in back propogation it is known as "W^O"
    Collective<t> queryWeights, keyWeights, valueWeights, outputWeights; // In the original paper "output projection weights" has same shape as the other three weight matrices.
                                                                        // In reality "output projection weights" should have the same shape projection weights for value input because these weights are multiplied with the output of product between attention scores and value weights

    Collective<t> attention_head_output, attention_head_outputs;

    try
    {                   
        // Get the dimensions of the input array 'ei'      
        dimensionOfArray = ei.getShape().getDimensionsOfArray();

        /* 
         * Divide the input 'ei' into equal slices for each attention head.
         * Modify the last dimension to split across attention heads.
         * Divide the column dimension by number of attention heads for equal partitioning
         */ 
              /*dimensionOfArray[dimensionOfArray.size() - 1] = ei.getShape().getNumberOfColumns() / numberOfAttentionHeads; // h in original paper*/                
              // Create dimension object with modified dimensions for slicing of 'ei' a.k.a encoder input
               /*dimension_ei_slice = DIMENSIONS(dimensionOfArray);*/

                //std::cout<< "OK = " << dimension.getNumberOfColumns() << ",  = " << dimension.getN() << std::endl;

                /*
                 * Initialize weight matrices if they haven't been initialized yet
                 * This lazy initialization creates the Q, K, V projection weights on first use
                 */
                if (queryWeights.getShape().getN() == 0 && keyWeights.getShape().getN() == 0 && valueWeights.getShape().getN() == 0 /*&& outputWeights.getShape().getN() == 0*/)
                {
                    // Set dimensions for full weight matrices (model_dim x model_dim)
                    dimensionOfArray[dimensionOfArray.size() - 1] = dimensionsOfTheModel; // d_model in original paper, ei.getShape().getNumberOfColumns();
                    dimensionOfArray[dimensionOfArray.size() - 2] = dimensionsOfTheModel; // d_model in original paper, ei.getShape().getNumberOfRows();
                    dimension_qkv_weights = DIMENSIONS(dimensionOfArray);
 
                    // Initialize Q, K, V weight matrices with random values
                    queryWeights = Numcy::Random::randn<t>(dimension_qkv_weights);
                    keyWeights = Numcy::Random::randn<t>(dimension_qkv_weights); 
                    valueWeights = Numcy::Random::randn<t>(dimension_qkv_weights); // Value weights can have fewer or more fetures than input features
 
                    /*
                     * Output projection weights
                     * W<sup>O</sup> has shape d_model$×$(d<sub>v</sub>·h)
                     * Since W<sup>V</sup> shape is no different than the other two projection weights (W<sup>Q</sup>, W<sup>K</sup>), the shape of W<sup>O</sup> will be same as W<sup>V</sup>
                     * We will not work on slices of these weights. This will be used as a right operand in a dot product operation, the other operand is concatenation of output of all (h many) attention heads
                     */
                    outputWeights = Numcy::Random::randn(dimension_qkv_weights);
                    
                    // Set dimensions for sliced weight matrices (per attention head)
                    dimensionOfArray[dimensionOfArray.size() - 1] = dimensionsOfTheModel / numberOfAttentionHeads; // d_q, d_k, d_v. d_q and d_k are interchangeable but d_v can be different.
                                                                                                                   // In the original paper, you would see d_k, d_k where d_q, d_k would have been used.  
                    dimension_qkv_slice = DIMENSIONS(dimensionOfArray);
                }
                
                // Iterate through all MultiHeadAttention modules in the linked list                
                while (current != NULL)
                {   
                    // AXIS_ROWS means we are slicing along and across rows vertically
                    // ---------------------------------------------------------------
                    // Extract a slice from input 'ei' starting at calculated position
                    // Each slice corresponds to one attention head's portion of the input                     
                            /*ei_slice = ei.slice(i*dimension_ei_slice.getNumberOfColumns(), dimension_ei_slice, AXIS_ROWS);*/

                    // Extract corresponding slices from Q, K, V weight matrices for this attention head
                    q_slice = queryWeights.slice(i*dimension_qkv_slice.getNumberOfColumns(), dimension_qkv_slice, AXIS_ROWS);
                    k_slice = keyWeights.slice(i*dimension_qkv_slice.getNumberOfColumns(), dimension_qkv_slice, AXIS_ROWS);
                    v_slice = valueWeights.slice(i*dimension_qkv_slice.getNumberOfColumns(), dimension_qkv_slice, AXIS_ROWS); 
                    
                    // AXIS_COLUMN means we are concatenating horizontally (along columns)
                    // -------------------------------------------------------------------
                    attention_head_output = MultiHeadAttention<t>::worker(ei, ei, ei, q_slice, k_slice, v_slice);

                    //std::cout<< "attention_head_ouput OK = " << attention_head_output.getShape().getNumberOfColumns() << ",  = " << attention_head_output.getShape().getNumberOfRows() << std::endl;

                    // AXIS_COLUMN means we are concatenating horizontally (along columns)
                    // -------------------------------------------------------------------
                    /*
                     * Concatenate individual attention head outputs along feature dimension
                     *
                     * Each attention head produces context vectors of shape (sequence_length × d_v)
                     * By concatenating along columns (feature axis), we combine the outputs from
                     * all h attention heads into a unified representation:
                     *
                     * Result shape: (sequence_length × h·d_v)
                     *
                     * This concatenated output preserves the diverse contextual information
                     * captured by each attention head, maintaining their unique specialized
                     * representations before the final linear projection.
                     */
                    attention_head_outputs = Numcy::concatenate(attention_head_outputs, attention_head_output, AXIS_COLUMN);

                    // AXIS_COLUMN means we are concatenating horizontally (along columns)
                    // -------------------------------------------------------------------
                    // Concatenate the current slice with previous slices along columns
                    // This builds up the complete processed output across all attention heads                    
                            /*ei_concatenated = Numcy::concatenate(ei_concatenated, ei_slice, AXIS_COLUMN);*/
                    
                    // AXIS_ROWS means we are slicing along and across rows vertically
                    // ---------------------------------------------------------------
                    // Update the weight matrices with the sliced portions
                    // This distributes different parts of the weight matrices to different attention heads
                    queryWeights.update(i*dimension_qkv_slice.getNumberOfColumns(), q_slice, AXIS_ROWS);
                    keyWeights.update(i*dimension_qkv_slice.getNumberOfColumns(), k_slice, AXIS_ROWS);
                    valueWeights.update(i*dimension_qkv_slice.getNumberOfColumns(), v_slice, AXIS_ROWS);
                                        
                    // Increment slice counter and move to next attention head
                    i = i + 1;
                    current = current->next;                                        
                }

                // Debug output to verify concatenated dimensions
                /*std::cout<< "Concatenated OK = " << ei_concatenated.getShape().getNumberOfColumns() << ",  = " << ei_concatenated.getShape().getNumberOfRows() << std::endl;*/                
                /*std::cout<< "attention_head_ouputs OK = " << attention_head_outputs.getShape().getNumberOfColumns() << ",  = " << attention_head_outputs.getShape().getNumberOfRows() << std::endl;*/

                /*for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < ei.getShape().getN(); k++)
                {
                    if (ei_concatenated[k] == ei[k])
                    {
                        std::cout<< "Mismatch at index " << k << ": ei_concatenated = " << ei_concatenated[k] << ", ei = " << ei[k] << std::endl;
                    }
                }*/

                /*for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < numberOfAttentionHeads; i++)
                {
                    ei_slice = ei.slice(i*dimension.getNumberOfColumns(), dimension, AXIS_ROWS);    

                    std::cout<< ei_slice.getShape().getN() << std::endl;

                    std::cout<< ei_slice.getShape().getDimensionsOfArray().size() << std::endl;
                }*/
            }
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayer<t>::forward(Collective<t>&, Collective<t>&, ENCODER_LAYER_NORM_POSITION_TYPE, bool) Error: ") + e.what());
            }

            //std::cout<< "Columns = " << ei.getShape().getNumberOfColumns() << ", Rows = " << ei.getShape().getNumberOfRows() << std::endl;

            //std::cout<< ei.getShape().getDimensionsOfArray()[ei.getShape().getDimensionsOfArray().size() - 1] << std::endl;
            //std::cout<< attentionMaskInputSequence.getShape().getDimensionsOfArray()[attentionMaskInputSequence.getShape().getDimensionsOfArray().size() - 1] << std::endl;

            //return Collective<t>{NULL, DIMENSIONS{0, 0, NULL, NULL}};

            // Return the concatenated result from all attention heads
            return /*ei_concatenated*/ attention_head_outputs;
        }
```
```C++
template <typename t = double>
struct MultiHeadAttention
{
    static Collective<t> worker(Collective<t>& Q, Collective<t>& K, Collective<t>& V, Collective<t>& w_q_slice, Collective<t>& w_k_slice, Collective<t>& w_v_slice) throw (ala_exception)
    {
        /*std::cout<< Q.getShape().getNumberOfColumns() << ", " << Q.getShape().getNumberOfRows() << std::endl;*
        /*std::cout<< w_q_slice.getShape().getNumberOfColumns() << ", " << w_q_slice.getShape().getNumberOfRows() << std::endl;*/
        
        Collective<t> Q_projected, K_projected, V_projected; 
     
        Collective<t> attention_scores, attention_weights, context_vector;
        // Scale factor for attention scores as defined in "Attention Is All You Need"
        // The scale factor 1/sqrt(d_k) prevents softmax saturation in the attention mechanism
        // when key dimension d_k is large, ensuring stable gradients during training
        t scaleFactor = 0;

        try 
        {
            // Transform these three inputs (Q, K, V) through linear projection for multi-head attention
            // 
            // This operation applies a learned (important aspect) linear transformation to the inputs (Q, K, V) 
            // using the attention head-specific weights w_q_slice, w_k_slice, w_v_slice. Each attention 
            // head learns distinct projection patterns that allow the model to focus on 
            // different aspects of the input sequence simultaneously
            //
            // Mathematically: Q_projected = Q · W<sub>i</sub><sup>Q</sup>
            //                 K_projected = K . W<sub>i</sub><sup>K</sup> 
            //                 V_projected = V . W<sub>i</sub><sup>V</sup>
            // Where:
            // - Q, K: input query and key matrices having shape = (sequence_length × d_model)
            // - V: input value matrix it may or may not have the same number of features as the Q, V matrices.  
            // - W<sub>i</sub><sup>Q/K</sup>: head-specific query and key weights having same shape = (d_model × <d_q, d_k>)
            // - W<sub>i</sub><sup>V</sup>: head-specific value weights having same shape = (number of featues × d_v) 
            // - Q_projected: transformed queries for head h (sequence_length × d_k)
            //
            // The projections enables each attention head to learn specialized query, key , value 
            // representations that capture different types of relationships and dependencies
            // within the sequences, a key mechanism behind the multi-head attention's
            // ability to process information in parallel and concurrently from multiple representation subspaces
            Q_projected = Numcy::dot(Q, w_q_slice);
            K_projected = Numcy::dot(K, w_k_slice);
            V_projected = Numcy::dot(V, w_v_slice);

            // We use the projected key dimension (number of features) because attention scores are computed as:
            // attention_scores = (Q_projected · trabspose(K_projected)) / sqrt(d_k)
            // where d_k is the dimension (number of features) of the projected key vectors
            scaleFactor =  1 / std::sqrt(static_cast<t>(K_projected.getShape().getNumberOfColumns()));

            // Compute scaled dot-product attention scores
            // 
            // 1. Calculate raw attention scores: Q_projected · transpose(K_projected)
            //    - Measures compatibility between each query and key pair
            //    - Results in matrix of shape (sequence_length × sequence_length)
            //
            // 2. Apply scale factor: Multiply by 1/sqrt((d_k))
            //    - Prevents softmax saturation when key dimension (number of features) d_k is large
            //    - Ensures stable gradients and effective training
            //    - As defined in "Attention Is All You Need" paper
            //
            // Result: Scaled attention scores ready for softmax normalization
            attention_scores = Numcy::dot(Q_projected, Numcy::transpose(K_projected));
            attention_scores = attention_scores * scaleFactor;

            attention_weights = Numcy::softmax(attention_scores);
            
            /*
             * Compute context vector: weighted sum of values based on attention distribution
             *
             * This operation creates a weighted sum of value vectors based on attention weights.
             * Each output position contains context gathered from all relevant input positions.
             * It represents the "context" that each query position attends to
             */
            context_vector = Numcy::dot(attention_weights, V_projected);
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("MultiHeadAttention::worker(Collective<&>) -> ") + e.what());
        }

        return context_vector;
    }
};
```