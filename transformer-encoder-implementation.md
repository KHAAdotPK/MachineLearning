```C++
/*
      transformer-encoder-decoder.md
      Written by, Q@khaa.pk		
 */
```

### Implementing Transformer Encoders in C++: From 'Attention is All You Need'

#### The encoder is composed of a stack of N=6 identical layers.
---
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

#### We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer​(x))(Post LN: Apply sublayer → Add residual → Normalize), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel=512. 

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



