```C++
/*
      transformer-encoder-implementation.md
      Q@khaa.pk		
 */
```

### Implementing Transformer Encoders in C++: From 'Attention is All You Need'
#### Written by, Sohail Qayum Malik
---

#### Project Overview

This series of documents provide a comprehensive analysis of a custom C++ transformer implementation, focusing on the complete pipeline from input sequence processing through encoder input/output preparation, decoder input/output preparation. The implementation represents a complete from-scratch build of the transformer architecture, including custom word embeddings, novel position encoding, and sophisticated build system architecture.

#### The encoder is composed of a stack of N=6 identical layers.
---
```TEXT
/*
    ==================== Encoder Architecture ====================
    
    	    Input Embeddings + Positional Encoding
        		             ↓           
	        ┌─────────────────────────────────────┐
	        │  Encoder Layer 1                    │
	        │  ┌──────────────────────────────┐   │
	        │  │   Multi-Head Self-Attention  │   │
	        │  └──────────────────────────────┘   │
	        │                ↓ (+ residual)       │
	        │           Layer Normalization       │
	        │                ↓                    │
	        │  ┌──────────────────────────────┐   │
	        │  │    Feed-Forward Network      │   │
	        │  └──────────────────────────────┘   │
	        │               ↓ (+ residual)        │
	        │       Layer Normalization           │
	        └─────────────────────────────────────┘
	            	        ↓
		            [Repeat 5 more times]
            		        ↓
        Encoded Output (ready for decoder or downstream task)
*/
```
```C++
/*
    EncoderLayerList represents a doubly linked list of encoder layers in a Transformer encoder stack.

    ==================== Purpose ====================
    In the Transformer architecture (as described in "Attention is All You Need"), the encoder is not a single unit but 
    a stack of *identical* encoder layers (usually 6 or 12). Each layer performs self-attention, followed by a feed-forward
    network, and each is wrapped with residual connections and layer normalization.

    Rather than using an array/vector of EncoderLayer objects, this implementation uses a doubly linked list to represent 
    a flexible and extendable chain of encoder layers.

    ==================== Structure ====================
    struct EncoderLayerList {
        EncoderLayer<t>* ptr;               // Pointer to the actual encoder layer (self-attention + FFN)
        EncoderLayerList<t>* next;          // Pointer to the next encoder layer in the stack
        EncoderLayerList<t>* previous;      // Pointer to the previous encoder layer in the stack
    };

    ==================== Benefits of Linked Structure ====================
    - **Flexibility in Construction**: Layers can be added or removed dynamically without resizing a contiguous array.
    - **Clear Navigation**: The 'previous' pointer allows easy backtracking (useful in training/debugging).
    - **Memory Management**: Decouples layer allocation from model management—layers can be individually allocated/freed.

    ==================== Usage Flow ====================
    1. Each node holds an `EncoderLayer` that implements its own `forward()` logic.
    2. During forward propagation through the encoder stack:
        - The input (a sequence of vectors) is passed to the first layer.
        - Its output is passed to the `next` EncoderLayerList node, and so on.
    3. This continues recursively or iteratively until the final encoder layer is reached.
    4. The output from the last EncoderLayer becomes the final encoded representation.

    ==================== Design Considerations ====================
    - Although vector-based storage might be more cache-friendly and efficient in real-time applications,
      the linked list structure reflects educational clarity and manual control, suitable for low-level
      or custom framework implementations like this one.
    - It is assumed that a separate managing class or model will hold a reference to the head of this list
      and control iteration through it.
    - In a training context, gradients can propagate backward using the `previous` pointers (manual backward pass).

    ==================== In Context ====================
    This list-based encoder stack directly reflects the architecture of the Transformer model's encoder block,
    where each EncoderLayer transforms the input to a richer representation by capturing dependencies between tokens.

    This structure is essential in building encoder-decoder models for tasks such as:
        - Machine translation
        - Text summarization
        - Question answering
        - Language modeling

    See EncoderLayer.hh for internal layer logic and attention mechanisms.    
 */
template <typename t /*= double*/> // Uncomment the default assignment of type and at compile time: warning C4348: 'DecoderLayer': redefinition of default parameter: parameter 1
class EncoderLayer;

template <typename t = double>
struct EncoderLayerList
{
    /*
        Transformer encoder layer
     */
    class EncoderLayer<t>* ptr; 

    struct EncoderLayerList<t>* next;
    struct EncoderLayerList<t>* previous;
};
```
```C++
        /*
            All arguments are "hyperparameters". Learn more about them in DOCUMENTS/hyperparameters.md
            @d_model, the dimension of the embedding space. Like Weights of NN determines the capacity and expressive power of the model.
            @num_layers, number of encoders. In the original paper about Transformers "Attention is all we need", six encoders were used.
            @num_heads, the number of atention heads. 
            @dropout_rate, it represents the probability of randomly "dropping out" or deactivating units (neurons) in a layer/encoder. Typically set between 0.1 and 0.5.
         */
        Encoder(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_layers, cc_tokenizer::string_character_traits<char>::size_type num_heads, t dropout_rate) : dimensionsOfTheModel(d_model), numberOfLayers(num_layers), numberOfAttentionHeads(num_heads), encoderLayerListHead(NULL), dropOutRate(dropout_rate)
        {
            if (dropout_rate < 0.0 || dropout_rate > 1.0)
            {
                dropout_rate = DEFAULT_DROP_OUT_RATE_HYPERPARAMETER;
                
                std::cerr << "Encoder::Encoder() Warning: Invalid dropout_rate provided (" << dropout_rate << "). " << "The dropout_rate must be between 0.0 and 1.0. " << "Using default value: " << DEFAULT_DROP_OUT_RATE_HYPERPARAMETER << "." << std::endl;
            }

            /*ENCODERLAYERLIST_PTR*/EncoderLayerList<t>* current = NULL; 
                        
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < numberOfLayers; i++)
            {                                
                if (current == NULL)
                {                    
                    current = /*new EncoderLayerList<t>();*/ reinterpret_cast</*ENCODERLAYERLIST_PTR*/EncoderLayerList<t>*>(cc_tokenizer::allocator<char>().allocate(sizeof(/*ENCODERLAYERLIST*/EncoderLayerList<t>)));
                    encoderLayerListHead = current;
                    current->previous = NULL;                    
                }
                else
                {                 
                    current->next = /*new EncoderLayerList<t>();*/ reinterpret_cast</*ENCODERLAYERLIST_PTR*/EncoderLayerList<t>*>(cc_tokenizer::allocator<char>().allocate(sizeof(/*ENCODERLAYERLIST*/EncoderLayerList<t>)));
                    current->next->previous = current;
                    current = current->next;
                }
                
                current->next = NULL; 
                /*
                    In a Transformer-based encoder (such as in BERT or GPT-like models), each encoder layer consists of multiple sublayers, typically:
                    1. Self-Attention Layer
                    2. Feedforward Layer
                    3. Layer Normalization (before or after these)
                 */     
                current->ptr = new EncoderLayer<t>(dimensionsOfTheModel, numberOfAttentionHeads, dropOutRate);                
            }                       
        }
```

#### Each layer has two sub-layers. 
---

- The first is a multi-head self-attention mechanism, 

```C++
	/*
    	    The encoder consists of many encoder layers.
	 */
	template <typename t = double>
	class EncoderLayer
	{       
	    Attention<t> attention;
```

- And the second is a simple, position-wise fully connected feed-forward network.

```C++
	/*
    	   The encoder consists of many encoder layers.
	 */
	template <typename t = double>
	class EncoderLayer
	{       
            Attention<t> attention;
	    EncoderFeedForwardNetwork<t> ffn; // Forward Feed Network
```

#### We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1].
---
 
```C++
	Collective<t> output, residual; /* residual: Store the output for the residual connection */
	EncoderLayerNormalization<t> /*norm1*/ attention_norm, /*norm2*/ ffn_norm; // Layer Normalization
```

#### That is, the output of each sub-layer is LayerNorm(x + Sublayer​(x))(Post LN: Apply sublayer → Add residual → Normalize), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel=512.
--- 

```C++
       if (norm_position == PreAttentionAndFeedForwardNetwork)
       {
          // Pre-LN for attention
          residual = attention_norm.forward(ei);  // Normalize first
          residual = attention.forward(residual, residual, residual/*, mask*/, attentionMaskInputSequence);
          output = ei + residual;  // Add residual connection
       }
       else if (norm_position == PostAttentionAndFeedForwardNetwork)
       {
          // Post-LN for attention
          residual = attention.forward(ei, ei, ei/*, mask*/, attentionMaskInputSequence);
          output = ei + residual;  // Add residual connection
          output = attention_norm.forward(output);  // Normalize after residual
       }
```

```C++
       if (norm_position == PreAttentionAndFeedForwardNetwork)
       {
          
          // Pre-LN for feed-forward network
          residual = ffn_norm.forward(output); // Layer norm before FFN
          residual = ffn.forward(residual);   // Apply FFN
          output = output + residual;         // Add residual                    
       }
       else if (norm_position == PostAttentionAndFeedForwardNetwork)
       {
	  // Post-LN for feed-forward network
          residual = ffn.forward(output);  // Apply FFN
          output = output + residual;     // Add residual
          output = ffn_norm.forward(output); // Layer norm after residual
       }
```

#### Example usage

```C++
    buildInputSequence(icp, iv, is, mask, attentionMaskInputSequence, W1, !ALLOW_REDUNDANCY); 
    buildPositionEncoding(pe, dt, dm, is, attentionMaskInputSequence /*mask*/, mntpl_input, sin_transformed_product, cos_transformed_product); 

    ei = pe + is;

    Encoder<t> encoder(ei.getShape().getNumberOfColumns(), DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER);                                 
    Collective<t> encoder_output = encoder.forward(ei, mask, attentionMaskInputSequence);          
```



