```
/*
	propogation.md
	Written by, Sohail Qayum Malik.
 */
```

**Propagation** refers to the process of calculating and updating model weights through forward and backward passes.

## __Forward__ ~~and __Backward__~~ propogation in `CBOW` and `Skip-gram` 
**Word2Vec** has two **word embedding algorithms**, the `Skip-gram` and `CBOW`. `Skip-gram` predicts context words for a center/target word, while `CBOW` predicts the center/target word for given context words.
#### Steps involved in Forward propogation are more or less similar in `CBOW` and `Skip-gram`
1. **Context Extraction**:
- Both `Skip-gram` and `CBOW` require identifying the context words. For `CBOW`, this involves averaging the embeddings of all context words, whereas, in `Skip-gram`, this would typically involve extracting the embedding for a single target word.
- __Example of context extraction in__ `Skip-gram`
```C++
/*
    Extract the corresponding word embedding from the weight matrix 𝑊1.
    The embedding for the center word is stored in the hidden layer vector h.
*/
double* h_ptr = cc_tokenizer::allocator<double>().allocate(W1.getShape().getNumberOfColumns());
// Loop through the columns of W1 to extract the embedding for the center word.
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W1.getShape().getNumberOfColumns(); i++)
{
    *(h_ptr + i) = W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + i];
}
Collective<double> h = Collective<double>{h_ptr, DIMENSIONS{W1.getShape().getNumberOfColumns(), 1, NULL, NULL}};
```
- __Example of context extraction in__ `CBOW`
```C++
/*
    For CBOW, h is the average of the embeddings of the context words. 
    This involves accessing the embeddings for all context words and averaging them.
    - OR -
    Compute the average of the embeddings of the context words to get a hidden representation, then use it to predict the center word.
 */
cc_tokenizer::string_character_traits<char>::size_type j = 0;
cc_tokenizer::string_character_traits<char>::size_type* ptr = cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().allocate(CBOW_WINDOW_SIZE*2);
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
{
    if ((*(pair->getLeft()))[i] >= INDEX_ORIGINATES_AT_VALUE)
    {
        ptr[j] = (*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE;
        j = j + 1;
    }
}
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
{
    if ((*(pair->getRight()))[i] >= INDEX_ORIGINATES_AT_VALUE)
    {
        ptr[j] = (*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE;
        j = j + 1;
    }
}
Collective<cc_tokenizer::string_character_traits<char>::size_type> context = Collective<cc_tokenizer::string_character_traits<char>::size_type>{ptr, DIMENSIONS{CBOW_WINDOW_SIZE*2, 1, NULL, NULL}};
/*
    Extract the corresponding word embeddings from the weight matrix 𝑊1, 
    average them and then store it in hidden layer.
 */
Collective<double> h = Numcy::mean(W1, context);
```
2. **Dot Product with Output Weights**:
- Both algorithms then perform a dot product between the hidden layer representation (h) and the output weight matrix (W2).
```C++
/*
    The dot product gives us the logits or unnormalized probabilities (u), which can then be transformed into probabilities using a softmax function
 */
Collective<E> u = Numcy::dot(h, W2);
```
This transformation step(`the dot product`) is crucial in both algorithms to map the hidden representation to the vocabulary space.

3. **Positive predicted probablities**:
```C++
        /*
            The resulting vector (u) is passed through a softmax function to obtain the predicted probabilities (y_pred). 
            The softmax function converts the raw scores into probabilities.
         */
        Collective<E> y_pred = softmax<E>(u);
```
    - In `Skip-gram`, this output represents the likelihood of each word being one of the context words for the given center word.
    - In `CBOW`, this output represents the likelihood of each word being the target word, given the context words.

## ~~__Forward__ and~~ __Backward__ propogation in `CBOW` and `Skip-gram` 
In `Skip-gram`, the objective is to predict the surrounding context words based on the center/target word, so during **backward propagation**, you compare the ~~positive~~ **predicted probabilities** (from **forward propagation**) to a **one-hot** encoded representation of the context words. This helps to update the weights based on how well the model predicted the actual context words given the center/target word.
```C++    
     /*
        Creating a One-Hot Vector, using Numcy::zeros with a shape of (1, vocab.numberOfUniqueTokens()).
        This creates a zero-filled column vector with a length equal to the vocabulary size
     */
    Collective<T> oneHot = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens(), 1, NULL, NULL});
    /*
        The following code block, iterates through the context word indices (left and right) from the pair object.
        For each valid context word index (i), it sets the corresponding element in the oneHot vector to 1.
        This effectively creates a one-hot encoded representation of the context words.
        - OR -
        In the one-hot vector, the position corresponding to the correct context word is set to 1, and all other positions are 0. 
     */        
    for (int i = SKIP_GRAM_WINDOW_SIZE - 1; i >= 0; i--)
    {       
        if (((*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens())
        {
            oneHot[(*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
        }
    }
    for (int i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
    {
        if (((*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens())
        {
            oneHot[(*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
        }        
    }

    /* 
        The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1, len(vocab) without redundency)

        Compute the gradient for the center word's embedding by subtracting the one-hot vector of the actual context word
        from the predicted probability distribution.
        `fp.predicted_probabilities` contains the predicted probabilities over all words in the vocabulary.
        `oneHot` is a one-hot vector representing the true context word in the vocabulary.
        The result, `grad_u`, is the error signal for updating the center word's embedding in the Skip-gram model.

        what is an error signal?
        -------------------------
        1. For the correct context word (where oneHot is 1), the gradient is (predicted_probabilities - 1), meaning the model's prediction was off by that much.
        2. For all other words (where oneHot is 0), the gradient is simply predicted_probabilities, meaning the model incorrectly assigned a nonzero probability to these words(meaning the model's prediction was off by that much, which the whole of predicted_probability for that out of context word).        
     */
    Collective<T> grad_u = Numcy::subtract<double>(fp.positive_predicted_probabilities, oneHot);

    /*
        The gradient (grad_u) used to update both the center word and the context word embeddings. 
     */
```
In `CBOW`, the objective is to predict the center/target word based on the surrounding context words, so during **backward propagation**, you compare the ~~positive~~ **predicted probabilities** (from **forward propagation**) to a **one-hot** encoded representation of the center/target word. This helps to update the weights based on how well the model predicted the actual center/target word for the given context words.
```C++
    /*
        Creating a One-Hot Vector, using Numcy::zeros with a shape of (1, vocab.numberOfUniqueTokens()).
        This creates a zero-filled column vector with a length equal to the vocabulary size
     */
    Collective<T> oneHot = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens(), 1, NULL, NULL});

    /*
        In the one-hot vector, the position corresponding to the center/target word is set to 1, and all other positions are 0.
     */
    oneHot[pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE] = 1;
    
    /* 
        The shape of `grad_u` is the same as `y_pred` (fp.predicted_probabilities) which is (1, len(vocab)).

        Compute the gradient for the center word embedding by subtracting the one-hot vector of the center/target word
        from the predicted probability distribution.
        `fp.predicted_probabilities` contains the predicted probabilities over all words in the vocabulary.
        `oneHot` is a one-hot vector representing the true center word in the vocabulary.
        The result, `grad_u`, is the error signal for updating the context words' embeddings in the CBOW model.

        What is an error signal?
        -------------------------
        1. For the correct center/target word (where `oneHot` is 1), the gradient is (predicted_probabilities - 1), meaning the model's prediction was off by that much.
        2. For all other words (where `oneHot` is 0), the gradient is simply `predicted_probabilities`, meaning the model incorrectly assigned a nonzero probability to these words.
     */
    Collective<T> grad_u = Numcy::subtract<double>(fp.positive_predicted_probabilities, oneHot);
    
    /*
        The gradient (grad_u) is used to update the embeddings of the context words, based on the error between the predicted center word and the true center word.
     */
```
**Weight Updates**: After computing the gradient (grad_u), lets see how the weights of both the input (`W1`) and output (`W2`) matrices are updated using [gradient descent](./GradientDescent.md).

1. `Skip-gram` backpropagation weight updates: After calculating the gradients (grad_u), the next step is to update both the input weight matrix (`W1`) and the output weight matrix (`W2`) using [gradient descent](./GradientDescent.md).
    - **Updating Output Weights (W2)**:
    You first compute `grad_W2`, which represents the gradient of the loss function with respect to the weights in `W2`. This is done by taking the outer product of the hidden layer activations (`fp.intermediate_activation`) and the gradient of the loss (`grad_u`).
    ```C++
    Collective<T> grad_W2 = Numcy::outer(fp.intermediate_activation, grad_u);
    ```
    This step calculates how much the output weights (`W2`) need to change to reduce the error in predicting the context words.
    - **Computing Gradient for Hidden Layer (grad_h) Update**:
    Next, you compute the gradient for the hidden layer (grad_h) using the transpose of W2. This is done by performing a dot product between grad_u and the transposed W2.
    ```C++
    Collective<T> W2_T = Numcy::transpose(W2);
    Collective<T> grad_h = Numcy::dot(grad_u, W2_T);    
    ```
    The result, `grad_h`, represents how much the error signal propagates back to the hidden layer, which is used to update the center word's embedding in the input weight matrix (`W1`).
    - **Updating Input Weights (W1)**:
    After computing the error signal (`grad_h`), you update the center word's embedding in `W1`. Here, you're iterating over the columns of `grad_W1` and adding the gradient values to the corresponding embedding of the center word.
    ```C++
    Collective<T> grad_W1 = Numcy::zeros(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});

    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_W1.getShape().getNumberOfColumns(); i++)
    {
        grad_W1[(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + i] += grad_h[i];
    }
    ```
    This process adjusts the input weights based on how well the model predicted the context words. The update rule typically involves subtracting the gradient (scaled by the `learning rate`) from the current weights.
    ```C++
    // grad_weights_input_to_hidden is grad_W1
    W1 -= bp.grad_weights_input_to_hidden * lr;
    // grad_weights_hidden_to_output is grad_W2
    Collective<t> W2_reshaped = Numcy::reshape(W2, bp.grad_weights_hidden_to_output);
    W2_reshaped -= bp.grad_weights_hidden_to_output * lr;
    // Update W2
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)
    {
        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getNumberOfColumns(); j++)
        {
            W2[i*W2.getShape().getNumberOfColumns() + j] = W2_reshaped[i*W2_reshaped.getShape().getNumberOfColumns() + j];\
        }
    }
    ```

    








