```C++
    outline_of_constructing_decoder_input_pipeline.md
    Written by, Sohail Qayum Malik
```

`"Readers should be aware that this article represents an ongoing project. The information and code contained herein are preliminary and will be expanded upon in future revisions."`

```C++
/* ************************************************************************************************************************************* */
/*                                      EXPLANAION OF decoder_input, decoder_mask STARTS HERE                                            */
/* ************************************************************************************************************************************* */ 

#define DECODER_INPUT_PAD_VALUE 0 // <PAD>
#define DECODER_INPUT_BEGINNING_OF_SEQUENCE 1 // <BOS> or <START>, <BOS> is Beginning Of Sequence
#define DECODER_INPUT_END_OF_SEQUENCE 2 // <EOS> or <END>
#define DECODER_INPUT_UNKNOWN_VALUE 3 // <UNK>
#define DECODER_INPUT_DIMMENSIONS 3 // Input dimensions, typically 3D tensor: [batch_size, sequence_length, d_model]
```

### Decoder Input Pipeline 

The decoder's forward method expects the following parameters:

- `input/decoder_input`: The input sequence to the decoder, typically a tensor of shape [batch_size, sequence_length, d_model].
- `output/encoder_output`: The output from the encoder, which provides context for the decoder. 
- `encoder_mask`: A mask to prevent the decoder from attending to certain positions in the encoder output.
- `decoder_mask`: A mask to prevent the decoder from attending to certain positions in its own input.

#### decoder_input and decoder_mask?

1. decoder_input (parameter 1):

    - This IS the target input (shifted right during training)
    - Contains the target sequence tokens that decoder should generate
    - During TRAINING: Target sequence shifted right with <START> token Example: Target = "Hello World" -> decoder_input = "<START> Hello World"
    - During INFERENCE: Previously generated tokens + current prediction
    - Shape: [batch_size, target_sequence_length, d_model]
    - Gets processed through masked self-attention (can't see future tokens)

        - **d_model** is a Hyperparameter:
            - Represents the dimensionality of the model's embeddings.
            - These are not pre-trained like Word2Vec they start as random numbers and are trained from scratch... typically small numbers from a Gaussian or uniform distribution).
            - Common values: 128, 256, 512, 1024 (depends on model size)
            - Higher d_model = more capacity, but also more computation
            - During training, these vectors are updated via backpropagation to capture meaningful semantic relationships.

2. decoder_mask (Look-ahead mask) (parameter 3):
    - Prevents decoder from attending to future tokens
    - Lower triangular matrix (causal mask)
    - Ensures autoregressive property during training
    - Shape: [target_seq_len, target_seq_len]
    - Example for sequence length 4:
```TEXT    
         [[1, 0, 0, 0],    # Token 0 can only see token 0
          [1, 1, 0, 0],    # Token 1 can see tokens 0,1
          [1, 1, 1, 0],    # Token 2 can see tokens 0,1,2
          [1, 1, 1, 1]]    # Token 3 can see tokens 0,1,2,3   
```          

STEP 1: Example Target Corpus:

```TEXT
    Sentence 1: "I love cats"
    Sentence 2: "I love dogs"
    Sentence 3: "I hate cats"
    Sentence 4: "You love programming"
    Sentence 5: "We hate programming"
```    
    
STEP 2: EXTRACT ALL UNIQUE WORDS:
Unique words found: {"I", "love", "cats", "dogs", "hate", "You", "We", "programming"}
    
STEP 3: CREATE VOCABULARY WITH SPECIAL TOKENS:
```TEXT    
    Token ID | Token        | Purpose
    ---------|--------------|------------------
    0        | "<PAD>"      | Padding (for batching)
    1        | "<START>"    | Start of sequence
    2        | "<END>"      | End of sequence
    3        | "<UNK>"      | Unknown words
    4        | "I"          | Regular vocabulary
    5        | "love"       | Regular vocabulary
    6        | "cats"       | Regular vocabulary
    7        | "dogs"       | Regular vocabulary
    8        | "hate"       | Regular vocabulary
    9        | "You"        | Regular vocabulary
    10       | "We"         | Regular vocabulary
    11       | "programming"| Regular vocabulary
```    
    
STEP 4: TOKENIZE ALL SENTENCES USING SAME VOCABULARY:

```TEXT
    Original Sentences -> Token IDs
    "I love cats"       -> [1, 4, 5, 6, 2]     // <START> I love cats <END>
    "I love dogs"       -> [1, 4, 5, 7, 2]     // <START> I love dogs <END>
    "I hate cats"       -> [1, 4, 8, 6, 2]     // <START> I hate cats <END>
    "You love programming" -> [1, 9, 5, 11, 2]  // <START> You love programming <END>
    "We hate programming"  -> [1, 10, 8, 11, 2] // <START> We hate programming <END>
```    
    
- NOTICE THE CONSISTENCY:

    - Word "I" ALWAYS gets token ID 4 (in sentences 1, 2, 3)
    - Word "love" ALWAYS gets token ID 5 (in sentences 1, 2, 4)
    - Word "cats" ALWAYS gets token ID 6 (in sentences 1, 3)
    - Word "programming" ALWAYS gets token ID 11 (in sentences 4, 5)
 
/*
    The start and end tokens (like <BOS> for beginning of sequence and <EOS> for end of sequence) are not related to batch size, they're related to sequence structure and the decoding process itself.
    Purpose of Start/End Tokens
    Start Token (<BOS>, <START>, etc.):
    - Signals to the decoder where to begin generating the output sequence.
    - During training, the decoder input is shifted right, meaning the first token is always the start token.
    - It helps the model understand that it should start generating from this point.
    - Provides initial context for the first token prediction
    - Essential for the autoregressive nature of decoding
    - Example: If the target sequence is "Hello World", the decoder input during training would be "<START> Hello World".
    
    End Token (<EOS>, <END>, etc.):
    - Signals to the decoder when to stop generating tokens.
    - During training, the decoder learns to predict this token when it has completed generating the sequence
    - It helps the model understand when to stop generating further tokens.
    - Example: If the target sequence is "Hello World", the decoder input during training would be "<START> Hello World <END>".
    - The model learns to predict the end token when it has completed generating the sequence.
    - It is crucial for tasks like text generation, where the model needs to know when to stop producing output.
    - Prevents the model from generating infinite sequences  
 
    Why Batch Size Doesn't Matter
    Whether you have:
        Batch size = 1: [<BOS> token1 token2 ... tokenN <EOS>]
        Batch size = 32: 32 sequences, each still needing [<BOS> ... <EOS>]

    Each sequence in the batch needs its own start/end tokens because:

    1. The decoder processes each sequence independently
    2. Each sequence needs to know its own boundaries
    3. The attention mechanism relies on these positional cues       
 */
/*
    DECODER MASK STRUCTURE:
    
    For sequence "I love cats" with tokens [1, 4, 5, 6, 2]:
    Position:  0(<START>)  1(I)  2(love)  3(cats)  4(<END>)
    
    Decoder Mask (Lower Triangular Matrix):
    ```
         0  1  2  3  4
    0 [  1  0  0  0  0 ]  # <START> can only see <START>
    1 [  1  1  0  0  0 ]  # I can see <START>, I
    2 [  1  1  1  0  0 ]  # love can see <START>, I, love
    3 [  1  1  1  1  0 ]  # cats can see <START>, I, love, cats
    4 [  1  1  1  1  1 ]  # <END> can see all previous tokens
    ```    
    Values: 1 = allowed to attend, 0 = masked (not allowed)
 */

```C++
/* ************************************************************************************************************************************* */
/*                                        EXPLANAION OF decoder_input, decoder_mask ENDS HERE                                            */
/* ************************************************************************************************************************************* */
```   
