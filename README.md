# Neural Network from Scratch (C++)

A neural network built entirely from scratch in C++, with no ML frameworks. The project is organized as two progressive milestones — starting from a hand-coded scalar XOR network and scaling up to a full matrix-based multi-class classifier trained on the Iris dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Dependencies — Installing Eigen](#dependencies--installing-eigen)
- [How to Compile and Run](#how-to-compile-and-run)
- [File-by-File Guide](#file-by-file-guide)
  - [Phase 1 — Scalar Network (root folder)](#phase-1--scalar-network-root-folder)
  - [Phase 2 — Matrix Network (phase2_matrix/)](#phase-2--matrix-network-phase2_matrix)
- [How the Complete Project Works](#how-the-complete-project-works)
- [The Iris Dataset](#the-iris-dataset)
- [Sample Results](#sample-results)

---

## Project Overview

This project teaches neural networks by building one from the ground up in C++, following a natural learning progression:

1. **Phase 1** — A minimal neural network using raw scalar arithmetic (no libraries). Trains on XOR, which requires a hidden layer and demonstrates that a single-layer perceptron is not enough.
2. **Phase 2** — A refactored, production-style neural network using the Eigen linear algebra library for matrix operations. Supports mini-batch training, CSV data loading, model saving/loading to binary files, and an interactive command-line prediction interface.

Every concept — forward pass, backpropagation, gradient descent, activation functions, loss functions — is implemented manually, not imported from a library.

---

## Folder Structure

```
neural-network-cpp/
│
├── phase1_scalar/               ← Phase 1: scalar XOR network
│   ├── xor_simple.cpp           ← Standalone XOR demo (no headers needed)
│   ├── activation.h             ← Sigmoid activation (scalar)
│   ├── layer.h                  ← Single-sample dense layer
│   ├── loss.h                   ← MSE scalar loss
│   ├── network.h                ← Network class (layer stack)
│   └── main.cpp                 ← XOR training entry point
│
├── phase2_matrix/               ← Phase 2: matrix/batch network
│   ├── activation.h             ← Sigmoid + ReLU (updated, reusable)
│   ├── layer_matrix.h           ← Batch-capable DenseLayerMatrix
│   ├── loss_matrix.h            ← MSE on Eigen matrices
│   ├── dataloader.h             ← CSV loader, normalizer, batch iterator
│   ├── network.h                ← NetworkMatrix with save/load support
│   ├── iris_dataset.csv         ← Iris dataset (150 samples, 4 features)
│   ├── main.cpp                 ← Interactive CLI (train new / load saved)
│   ├── train_iris.cpp           ← Standalone Iris training script
│   ├── test_iris.cpp            ← Interactive Iris predictor
│   ├── test_batch.cpp           ← Tests batch forward pass
│   ├── test_eigen.cpp           ← Tests Eigen matrix setup
│   ├── test_network.cpp         ← Tests full forward/backward/update
│   ├── test_save_load.cpp       ← Tests model serialization
│   └── test_training.cpp        ← Tests XOR with batch backprop
│
└── README.md
```

> **Note on activation.h**: There are two versions. The one in `phase1_scalar/` is the original scalar version. The one in `phase2_matrix/` is the updated version that also includes ReLU and is used across all Phase 2 files.

---

## Dependencies — Installing Eigen

Phase 2 uses [Eigen](https://eigen.tuxfamily.org/), a header-only C++ library for matrix math. "Header-only" means you just download the headers and point your compiler at them — no building or linking required.

### Step 1 — Download Eigen

Go to https://eigen.tuxfamily.org/index.php?title=Main_Page and download the latest stable release (a `.zip` or `.tar.gz` file).

Alternatively, you can clone it with Git:
```bash
git clone https://gitlab.com/libeigen/eigen.git
```

### Step 2 — Extract and place it

Extract the archive. You will get a folder like `eigen-3.4.0/` containing a subfolder called `Eigen/`.

You can place this anywhere on your system. A common choice is to put it alongside your project:

```
neural-network-cpp/
├── eigen-3.4.0/        ← put it here (outside phase1/phase2 folders)
│   └── Eigen/
├── phase1_scalar/
├── phase2_matrix/
└── README.md
```

### Step 3 — Tell the compiler where to find it

When compiling, use the `-I` flag to point to the folder that **contains** the `Eigen/` directory:

```bash
g++ -I../eigen-3.4.0 your_file.cpp -o your_program
```

That's it. The `#include <Eigen/Dense>` in the source files will resolve correctly.

### Checking it works

Compile and run `test_eigen.cpp`:
```bash
cd phase2_matrix
g++ -I../../eigen-3.4.0 test_eigen.cpp -o test_eigen
./test_eigen
```

You should see a matrix printed to the console. If it compiles cleanly, Eigen is set up correctly.

---

## How to Compile and Run

### Phase 1 — XOR scalar network

```bash
cd phase1_scalar

# Standalone demo (no headers needed beyond the standard library)
g++ xor_simple.cpp -o xor_simple
./xor_simple

# Full network via header files
g++ main.cpp -o main
./main
```

### Phase 2 — Matrix network (requires Eigen)

Replace `../../eigen-3.4.0` with the actual path to your Eigen folder.

```bash
cd phase2_matrix

# Interactive CLI: train a new model or load a saved one
g++ -I../../eigen-3.4.0 main.cpp -o main
./main

# Train on Iris and print accuracy
g++ -I../../eigen-3.4.0 train_iris.cpp -o train_iris
./train_iris

# Interactive Iris predictor (enter flower measurements)
g++ -I../../eigen-3.4.0 test_iris.cpp -o test_iris
./test_iris

# Unit tests
g++ -I../../eigen-3.4.0 test_eigen.cpp -o test_eigen && ./test_eigen
g++ -I../../eigen-3.4.0 test_batch.cpp -o test_batch && ./test_batch
g++ -I../../eigen-3.4.0 test_training.cpp -o test_training && ./test_training
g++ -I../../eigen-3.4.0 test_network.cpp -o test_network && ./test_network
g++ -I../../eigen-3.4.0 test_save_load.cpp -o test_save_load && ./test_save_load
```

### Recommended compiler flags for better performance

```bash
g++ -O2 -std=c++17 -I../../eigen-3.4.0 main.cpp -o main
```

---

## File-by-File Guide

### Phase 1 — Scalar Network (root folder)

#### `xor_simple.cpp`
The most foundational file in the project. A complete neural network written in a single `.cpp` file with no headers, no classes, and no libraries — just raw variables and loops. It manually implements:
- Random weight initialization
- Sigmoid activation and its derivative
- Forward pass for a 2→2→1 network
- Backpropagation with batch gradient accumulation
- Weight update via gradient descent

This is the "show your work" version — every single step is visible and explained. Great for understanding what all the abstraction in Phase 2 is actually doing underneath.

**Task**: Learns the XOR function (0,0→0 / 0,1→1 / 1,0→1 / 1,1→0).

---

#### `activation.h` (Phase 1 version)
Defines the `Activation` abstract base class with two pure virtual methods: `apply(x)` and `derivative(x)`. Two concrete implementations:
- `Sigmoid`: `f(x) = 1 / (1 + e^-x)`, derivative = `s * (1 - s)` where s = sigmoid(x)
- `ReLU`: `f(x) = max(0, x)`, derivative = 1 if x > 0 else 0

Uses C++ polymorphism — you can pass any activation to a layer via `shared_ptr<Activation>`.

---

#### `layer.h`
Implements `DenseLayer`, a single fully-connected layer that operates on one sample at a time (scalar/vector, not batched). Stores:
- Weight matrix `W` (out_size × in_size)
- Bias vector `b`
- Cached inputs and pre-activation values for backprop
- Computed gradients `dW` and `db`

Methods: `forward(input)`, `backward(grad_output)`, `update(learning_rate)`.

---

#### `loss.h`
Defines the `Loss` abstract base class with `value(predicted, target)` and `gradient(predicted, target)` for scalar values. Implements `MSE` (mean squared error): `loss = (pred - target)^2`, gradient = `pred - target`.

---

#### `network.h` (Phase 1 version)
The `Network` class holds a vector of `DenseLayer` objects and a `Loss` function. Provides:
- `add_layer(layer)` — builds the architecture layer by layer
- `predict(input)` — runs the forward pass
- `train_epoch(inputs, targets, lr, batch_size, shuffle)` — one full training epoch

---

#### `main.cpp` (Phase 1)
Entry point for the Phase 1 network. Creates a 2→4→1 Sigmoid network, trains it on XOR for 2000 epochs, and prints predictions. The expected output is something like: `0,0 → 0.03`, `0,1 → 0.97`, `1,0 → 0.97`, `1,1 → 0.04`.

---

### Phase 2 — Matrix Network (`phase2_matrix/`)

#### `activation.h` (Phase 2 version)
Same design as Phase 1 but fully compatible with the matrix-based pipeline. The same `Sigmoid` and `ReLU` classes are used here. The key difference is that `layer_matrix.h` calls `activation->apply(x)` and `activation->derivative(x)` element-wise inside matrix loops.

---

#### `layer_matrix.h`
The core of Phase 2. `DenseLayerMatrix` is a batch-capable dense layer using Eigen matrices. Input is a `(batch_size × input_size)` matrix; output is `(batch_size × output_size)`.

**Forward pass**: `Z = X * W^T + b`, then `A = activation(Z)` element-wise. Caches `X`, `Z`, and `A` for backprop.

**Backward pass**: Given `dL/dA` from the next layer:
1. `dZ = dL/dA ⊙ activation'(Z)` (element-wise)
2. `dW = (dZ^T × X) / batch_size`
3. `db = mean of each column of dZ`
4. Returns `dZ × W` to pass upstream

**Update**: `W -= lr * dW`, `b -= lr * db`

---

#### `loss_matrix.h`
`MSEMatrix` computes MSE loss and its gradient over a full batch at once. Input: two `(batch_size × output_size)` matrices. Returns a scalar loss or a gradient matrix of the same shape.

---

#### `dataloader.h`
The most complex single file in the project. `DataLoader` handles everything between a raw CSV file and training-ready Eigen matrices:

- Reads and parses CSV (comma-separated, any number of columns)
- Auto-detects or accepts user-specified feature/label columns
- One-hot encodes string labels into a Y matrix
- Shuffles and splits data into train/validation sets
- Min-max normalizes features using train-set statistics (and applies the same normalization to validation to prevent data leakage)
- Stores `min_vals` and `ranges` so new samples can be normalized at prediction time
- Provides `get_next_batch()` for mini-batch iteration

Two constructors: auto-detect (assumes last column is the label) and manual column specification.

---

#### `network.h` (Phase 2 version)
`NetworkMatrix` holds a vector of `DenseLayerMatrix` layers and orchestrates training. Key additions over Phase 1:
- `set_normalization(min_vals, ranges)` — stores normalization params so they can be saved with the model
- `set_class_labels(labels)` — stores string class names
- `save(filename)` — serializes the entire model to a binary `.bin` file: input/output sizes, normalization vectors, class labels, and all layer weights/biases
- `load(filename)` — reconstructs the full model from a `.bin` file

The binary format is documented in-code. A saved model is fully self-contained — you can load it without knowing the original dataset.

---

#### `iris_dataset.csv`
The classic Iris dataset: 150 samples (149 in the file, as the header row was removed), 4 features (sepal length, sepal width, petal length, petal width in cm), and 3 classes (Iris-setosa, Iris-versicolor, Iris-virginica).

---

#### `main.cpp` (Phase 2 — interactive CLI)
The most user-facing file. Presents a menu:
1. **Train a new model** — prompts for dataset path, train/val split, batch size, architecture (number and size of hidden layers), learning rate, and epochs. Trains the network, prints per-epoch validation accuracy, shows sample predictions, and offers to save the model.
2. **Load a saved model** — lists all `.bin` files in the current directory, lets you pick one, and loads it with its normalization and class labels intact.

After either option, drops into a **prediction loop** where you type feature values and get an instant class prediction.

---

#### `train_iris.cpp`
A clean, self-contained Iris training script. No user interaction. Loads the CSV, builds a 4→8→3 Sigmoid network, trains for 350 epochs with mini-batches, and prints train loss + validation accuracy every 10 epochs. Good for quick experimentation.

---

#### `test_iris.cpp`
Trains the same 4→8→3 network then enters an interactive loop where you enter four flower measurements (in cm) and get back the predicted species with confidence percentages.

---

#### `test_batch.cpp`
Verifies that `DenseLayerMatrix` correctly handles a batch of 4 samples. Creates a 2→4→1 network and runs the full XOR batch through it, printing the hidden and output matrices. No training — just testing the forward pass shape.

---

#### `test_eigen.cpp`
Minimal file. Creates a 2×3 matrix and prints it. The only purpose is to confirm that Eigen is installed and the compiler can find the headers. Always compile this first when setting up on a new machine.

---

#### `test_network.cpp`
Tests `NetworkMatrix` end-to-end: creates a 4→8→3 network, loads Iris data, runs forward pass, computes MSE gradient, runs backward pass, and calls update. Confirms no crashes or shape mismatches.

---

#### `test_save_load.cpp`
Trains a 4→8→3 network on Iris for 500 epochs, saves it to `iris_model.bin`, creates a brand-new `NetworkMatrix`, loads the file, and runs the same validation batch through both networks. Checks that every output value is identical to 10 decimal places.

---

#### `test_training.cpp`
Matrix-based XOR training test. Uses `DenseLayerMatrix` directly (not `NetworkMatrix`) to train a 2→4→1 network on XOR for 10,000 epochs with batch backprop. Prints loss every 500 epochs and final predictions. A good sanity check that backprop is working correctly before moving to real data.

---

## How the Complete Project Works

### The Neural Network Pipeline

```
Raw CSV  →  DataLoader  →  Batches  →  Network (forward)  →  Loss
                                              ↑                  ↓
                                         update(lr)  ←  backward(gradient)
```

**1. Data loading**: `DataLoader` reads the CSV, one-hot encodes labels, shuffles, splits 80/20 into train/val, and normalizes features to [0,1] using min-max scaling computed from the training set only.

**2. Forward pass**: Each mini-batch (a matrix of shape `batch_size × features`) flows through the layers. In each `DenseLayerMatrix`: multiply by weights, add bias, apply activation. The output of the last layer is a `batch_size × num_classes` probability matrix.

**3. Loss**: `MSEMatrix` computes the mean squared error between predictions and one-hot targets. This tells us how wrong the network is.

**4. Backward pass**: Starting from the loss gradient, each layer computes: how much did each weight contribute to the error? This is done via the chain rule. The gradient flows backward through every layer.

**5. Weight update**: Each layer subtracts `learning_rate × gradient` from its weights. Over thousands of iterations, the weights converge to values that minimize the loss.

**6. Evaluation**: On the held-out validation set, we run only the forward pass, find the highest-confidence output neuron (argmax), and compare to the true label.

### What "from scratch" means

Every component in this list was written manually:
- Matrix multiplication (via Eigen, but the layer logic is manual)
- Activation functions and their derivatives
- Backpropagation chain rule through each layer
- Mini-batch gradient accumulation and averaging
- Feature normalization
- CSV parsing and one-hot encoding
- Binary model serialization

Nothing from PyTorch, TensorFlow, or any ML library was used.

---

## The Iris Dataset

The Iris dataset is one of the most well-known classification benchmarks. It contains measurements of 150 iris flowers across three species:

| Feature | Description |
|---------|-------------|
| Sepal length | Length of the outer leaf (cm) |
| Sepal width | Width of the outer leaf (cm) |
| Petal length | Length of the inner petal (cm) |
| Petal width | Width of the inner petal (cm) |

The three classes are `Iris-setosa`, `Iris-versicolor`, and `Iris-virginica`. Setosa is linearly separable from the other two; versicolor and virginica overlap slightly, making this a good test of a multi-layer network.

---

## Sample Results

Running `train_iris.cpp` with default settings (4→8→3, lr=0.5, 350 epochs):

```
Epoch 0   | Train Loss: 0.22 | Val Loss: 0.23 | Val Accuracy: 33.33%
Epoch 50  | Train Loss: 0.07 | Val Loss: 0.08 | Val Accuracy: 76.67%
Epoch 100 | Train Loss: 0.04 | Val Loss: 0.05 | Val Accuracy: 90.00%
Epoch 200 | Train Loss: 0.02 | Val Loss: 0.03 | Val Accuracy: 93.33%
Epoch 350 | Train Loss: 0.01 | Val Loss: 0.02 | Val Accuracy: 96.67%

Final Validation Accuracy: 96.67%
```

Results vary slightly each run due to random weight initialization and shuffling.
