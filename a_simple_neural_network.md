# Understanding a Simple Neural Network: Step by Step

In this article, we’ll walk through a basic **neural network** with one hidden layer. We’ll keep the explanation simple, using small numbers for easier understanding. This kind of network is used in **machine learning (ML)** to make predictions from input data.

### Our Neural Network Structure

Let’s say we have:
- **4 samples (data points)**  
- **Each sample has 16 input features**
- **A hidden layer with 32 neurons**
- **An output layer with 8 neurons**

---

## Step 1: Input to Hidden Layer

We start with our input data:

- `input` has shape **[4][16]** — 4 samples, each with 16 features.

We also have weights connecting the input to the hidden layer:

- `W1` has shape **[32][16]** — 32 hidden neurons, each with 16 weights.

To multiply the input with the weights, we need to transpose `W1` (flip its rows and columns). So:

- `W1^T` becomes **[16][32]**

Now we do the matrix multiplication:

hidden = input × W1^T


This gives us:

- `hidden` has shape **[4][32]** — for each of the 4 samples, we get 32 hidden values (also called activations).

We also add a bias `b1[32]` to each row. Bias helps shift the activations and gives the model more flexibility.

---

## Step 2: Hidden to Output Layer

Now we pass the hidden layer output to the next layer — the output layer.

- `hidden` is **[4][32]**
- `W2` is **[8][32]** — 8 output neurons, each connected to all 32 hidden outputs.

We take the transpose of `W2`:

- `W2^T` is **[32][8]**

Then we multiply:

output = hidden × W2^T


This gives:

- `output` is **[4][8]** — 4 output vectors, each with 8 values (one for each output neuron).

We also add a bias `b2[8]` to each output.

---

## Summary (in C-style terms)

| Name    | Shape     | Meaning                                 |
|---------|-----------|-----------------------------------------|
| input   | [4][16]   | 4 samples, each with 16 input features  |
| W1      | [32][16]  | Weights for 32 hidden neurons           |
| b1      | [32]      | Biases for hidden layer                 |
| hidden  | [4][32]   | Output of hidden layer                  |
| W2      | [8][32]   | Weights for 8 output neurons            |
| b2      | [8]       | Biases for output layer                 |
| output  | [4][8]    | Final output predictions                |

---

## Final Notes

- The **forward pass** means moving data from input → hidden → output.
- Each layer multiplies the input by weights and adds bias.
- Activation functions (like ReLU) are usually applied after each layer, but we didn’t include them here to keep things simple.

This is the basic idea behind many deep learning models — just with more layers and data.


![NN](https://github.com/KHAAdotPK/MachineLearning/blob/main/nn.png)