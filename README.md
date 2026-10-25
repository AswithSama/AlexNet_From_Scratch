<div align="center">

# AlexNet From Scratch

### Understanding convolutional neural networks from first principles

**A research-oriented implementation study of AlexNet — from hand-written tensor operations to a trained image classifier.**

`Computer Vision` · `Convolutional Neural Networks` · `NumPy` · `TensorFlow / Keras` · `Deep Learning Fundamentals`

[Explore the Mathematical Deep Dive](DEEPDIVE.md) · [Open the Main Notebook](AlexNet_Tf_Final.ipynb) · [Original AlexNet Paper](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

</div>

---

## The idea behind this project

**What actually happens between an image entering a neural network and a class prediction coming out?**

A few lines of TensorFlow can define a convolutional neural network. That abstraction is useful, but it can also hide what makes a network work: sliding kernels, dot products, activation functions, pooling windows, parameter updates, the chain rule, and the gradual construction of visual features.

This project approaches AlexNet as something to **understand, reconstruct, and investigate**, rather than simply import and train. We explored a landmark CNN architecture through two complementary approaches:

1. **First-principles implementation:** write the important layer computations with NumPy, inspect how data moves through them, and assemble an AlexNet-inspired forward pass.
2. **Framework-based implementation:** construct an AlexNet-style network using TensorFlow/Keras, train it on a curated 16-class ImageNet subset, and examine what the model learns — and where it struggles to generalize.

The central goal was to connect **mathematics, calculus, and intuition** with executable code. Understanding *why* convolution, nonlinearities, pooling, normalization, and gradients behave as they do provides a foundation for understanding other deep neural networks, not just AlexNet.

![Two complementary ways we studied AlexNet: fundamental operations and model training](docs/assets/two_approaches.png)

> **Scope:** The custom NumPy work demonstrates forward computations, not a complete manually trained backpropagation system. The reported classification experiments belong to the separate TensorFlow/Keras model. This is intentionally an implementation-and-understanding study, not an end-to-end production classifier.

## Research questions

| Question | How we investigated it |
| --- | --- |
| How does a network extract information from pixels? | Implemented multi-channel sliding-window convolution and inspected feature-map dimensions. |
| Why do CNNs need nonlinearities? | Studied and applied ReLU so that stacked layers can represent nonlinear relationships. |
| How do spatial dimensions shrink without losing all useful information? | Implemented max pooling and traced the spatial shape after each stage. |
| How do activations remain numerically manageable? | Implemented local response normalization and explored a separate batch-normalization example. |
| How are visual features translated into class scores? | Implemented dense-layer matrix multiplication and assembled an AlexNet-inspired forward pass. |
| What changes when we train the architecture? | Built a TensorFlow CNN with SGD, regularization, callbacks, and 16-class evaluation. |
| Does learning the training images mean the model generalizes? | Compared recorded training and validation accuracy across 50 epochs. |

## What we implemented

### Approach 1 — Looking inside the network with NumPy

The first approach deliberately exposes operations that a deep-learning library would normally execute for us.

| Component | First-principles implementation | Concept made tangible |
| --- | --- | --- |
| **Convolution (`Conv2D`)** | Randomly initializes kernels; slides each filter over a multi-channel image; multiplies and sums local values | Local receptive fields, shared weights, channels, strides, padding |
| **Max pooling (`MaxPooling2D`)** | Scans spatial windows and keeps the largest activation | Downsampling and local activation selection |
| **Dense layer (`Dense`)** | Initializes weights and biases; applies matrix multiplication | Weighted sums, learned projections, output dimensions |
| **Local response normalization** | Computes channel-neighborhood energy and rescales activations | Normalization across nearby feature channels |
| **Batch normalization study** | Separate NumPy notebook with batch statistics and train/evaluation behavior | Mean, variance, scale/shift, running statistics |
| **AlexNet-inspired forward pass** | Connects custom layers into five convolution stages, pooling, and fully connected layers | How many simple operations form a deep architecture |

The core custom arithmetic is written in **NumPy**. The assembled example also uses TensorFlow operations for ReLU, dropout, and softmax; it is therefore a *hybrid forward-pass demonstration*, not a claim that every operation was written without a framework.

### A small convolution example

Think of a convolution filter as a small grid of weights. It looks at one region of the image, computes a weighted sum, moves across the image, and repeats. Different filters can respond to different visual patterns when trained.

![Worked example of a sliding convolution kernel producing a feature map](docs/assets/convolution_intuition.png)

In the included implementation, a `227 × 227 × 3` example image passed through **96 filters of size `11 × 11` with stride `4`** produces a `55 × 55 × 96` output. Watching this tensor change shape helps explain how a CNN builds a representation of an image.

The standalone layer studies live in [`Inbuilt_Modules_Vs_Custom2/`](Inbuilt_Modules_Vs_Custom2/); the assembled sample is also in [`AlexNet_Tf_Final.ipynb`](AlexNet_Tf_Final.ipynb).

### Approach 2 — Training an AlexNet-style model with TensorFlow

The second approach asks a different question: once the architecture is defined, **how do its parameters respond to data and optimization?**

We constructed a Keras `Sequential` model with:

- **Five convolutional layers** using `96 → 256 → 384 → 384 → 256` filters.
- **ReLU** activation and **batch normalization** at each convolution stage.
- **Three max-pooling operations**, after convolution stages 1, 2, and 5.
- **Two 4,096-unit fully connected layers** with `0.5` dropout.
- **A 16-class softmax head** adapted to our ImageNet subset.

![Architecture and exact tensor shapes of the TensorFlow AlexNet-style implementation](docs/assets/alexnet_architecture.png)

This is **AlexNet-inspired, not an exact reproduction of the 2012 paper**: for example, the trained implementation uses batch normalization in place of the original local response normalization, takes `224 × 224` inputs, and predicts 16 rather than 1,000 classes. The model summary in the notebook reports **46,818,064 total parameters**.

The custom demonstration and the trained model share the architectural *ideas*, but they serve different learning purposes.

## The dataset: a focused ImageNet experiment

We used the `Imagenet_mini/` collection included with the project: **16 categories**, comprising fish and bird classes.

![Representative images from the included 16-class ImageNet subset](docs/assets/imagenet_mini_gallery.jpg)

| Item | Recorded / archived count |
| --- | ---: |
| Categories | 16 |
| Images under `Imagenet_mini/train/` | 513 |
| Training subset selected by Keras | 462 |
| Validation subset selected by Keras | 51 |
| Additional images under `Imagenet_mini/test/` | 56 |
| Image size used for framework training | `224 × 224 × 3` |
| Training batch size | 2 |

The experiment obtains its **training and validation splits from the `train/` directory**, using `validation_split=0.1`, `seed=123`, and `subset="training"` / `"validation"`. Although a separate `test/` directory is present in the archive, the final notebook does **not** report an evaluation against it.

The preprocessing investigation included resizing, training-dataset mean computation and subtraction, horizontal reflection, and a brightness-adjustment function. In the **active final data pipeline**, normalization and horizontal reflection are used; brightness adjustment is commented out. The reflection runs on both training and validation data, an experimental limitation discussed in the [Deep Dive](DEEPDIVE.md#10-experimental-limitations-and-interpretation).

<details>
<summary><strong>View image previews captured directly from the notebook</strong></summary>

**Examples after loading and resizing**

![Sample resized images recorded in the original notebook](docs/assets/notebook_input_samples.png)

**Examples after the notebook's active preprocessing pipeline**

![Saved preview from the original notebook after preprocessing](docs/assets/notebook_augmented_samples.png)

The second plot is retained exactly as recorded in the notebook. Because the notebook converts mean-subtracted tensors directly to `uint8` for display, its appearance should not be interpreted as a faithful color rendering of the normalized floating-point tensors.

</details>

## Experiment setup

The final framework experiment uses these choices, taken from the notebook:

| Setting | Configuration |
| --- | --- |
| Architecture | Five-convolution AlexNet-style CNN |
| Optimizer | SGD, with weight decay `0.0005` |
| Starting learning rate | `0.01` |
| Loss | Categorical cross-entropy |
| Maximum epochs | 50 |
| Batch size | 2 |
| Validation allocation | 10% of the `train/` folder |
| Regularization | Batch normalization, dropout `0.5`, weight decay |
| Learning-rate adjustment | `ReduceLROnPlateau`, monitoring validation loss |
| Checkpoints | Per-epoch weight checkpoint files |
| Early stopping | `EarlyStopping`, monitoring training loss |

## Recorded results and what we learned

The original notebook contains a full 50-epoch training log followed by evaluations of the resulting model. These are **archived run results**, not claims from a new reproduction.

| Final evaluation | Accuracy | Loss |
| --- | ---: | ---: |
| Training subset | **81.82%** | `0.7818` |
| Validation subset | **23.53%** | `2.6801` |

The highest **epoch-end validation accuracy recorded during training** was **27.45% at epoch 31**. It is distinct from the final evaluation, which reported 23.53%.

![Training and validation learning curves reconstructed from the notebook log](docs/assets/training_curves.png)

The curves teach an important lesson: **being able to fit the training images is not the same as learning to classify unseen images well.** The training/validation gap is consistent with substantial overfitting and limited generalization on this very small dataset. It does not establish the exact cause of the gap, and no independent test-set accuracy is reported.

Rather than presenting the experiment as a benchmark result, we treat it as a study of **capacity, optimization, regularization, limited sample size, and generalization**. This connects the abstract mathematical ideas of a CNN with observable training behavior.

The per-epoch measurements are also provided in [`training_history_from_notebook.csv`](training_history_from_notebook.csv), transcribed programmatically from the saved notebook output for inspection and plotting.

## The deeper intuition

Some of the most useful insights from working through this project were:

1. **A convolutional feature map is just structured linear algebra.** Its expressive power comes from repeated local operations and the learned values of shared filters.
2. **Shapes are part of the algorithm.** Kernel size, stride, padding, and pooling control how much spatial structure survives from one layer to the next.
3. **Nonlinearity is essential.** Without activation functions, stacking linear transformations would still be a linear transformation.
4. **The chain rule connects prediction error to parameters.** Even though the custom branch focuses on forward propagation, studying derivatives explains how those same computations can be optimized during training.
5. **An architecture can be understood independently of achieving a high score.** The modest validation result is itself useful evidence about small-data deep-learning experiments.

For worked formulas, tensor shapes, parameter calculations, pseudocode, and discussion of what was actually written versus what Keras performed automatically, see **[DEEPDIVE.md](DEEPDIVE.md)**.

## Project structure

```text
AlexNet_From_Scratch/
├── AlexNet_Tf_Final.ipynb              # Custom forward pass + TensorFlow experiment
├── AlexNet_Tf_Final_Doc.html          # Exported notebook documentation
├── AlexNetUnderstandingNotes.pdf      # Original research/understanding notes
├── Inbuilt_Modules_Vs_Custom2/
│   ├── Conv2D.ipynb                   # Convolution, pooling, dense, LRN, assembly
│   ├── MaxPooling2D.ipynb             # Standalone pooling study
│   ├── BatchNormalization.ipynb       # Standalone batch normalization study
│   └── Local Response Normalization.ipynb
├── Imagenet_mini/
│   ├── train/                         # 16 class folders (513 images)
│   ├── test/                          # 16 class folders (56 images)
│   └── map_clsloc.txt
├── References/                       # Papers and background resources
├── Group_contribution.txt            # Collaborative project contributions
├── docs/assets/                      # Figures and notebook images for this README
├── training_history_from_notebook.csv
├── README.md
└── DEEPDIVE.md
```

## Exploring the implementation

The principal source of truth is the **notebook**, not a packaged training CLI. The following commands provide a starting point for exploring it in an isolated environment:

```bash
# From the repository root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib tensorflow jupyter
python -m jupyter notebook AlexNet_Tf_Final.ipynb
```

Run the notebook **section by section**, starting with the layer demonstrations. The full TensorFlow training portion is computationally heavier and attempts to save checkpoint files. The repository does not provide pinned dependencies, an automated reproduction script, or the saved checkpoints from the recorded experiment; environment compatibility and numerical outcomes may vary.

For the smaller mathematical examples, open the notebooks in [`Inbuilt_Modules_Vs_Custom2/`](Inbuilt_Modules_Vs_Custom2/). These are useful for following one layer at a time.

## What this study covers — and what it does not

**Implemented / investigated:** manual NumPy convolution, pooling, dense layers, local response normalization, separate batch-normalization exploration, custom AlexNet-inspired forward pass, TensorFlow training architecture, preprocessing, optimization configuration, callbacks, and recorded learning behavior.

**Not claimed:** NumPy-based backpropagation or end-to-end manual training; a PyTorch training implementation; full ImageNet reproduction; evaluation on the archived separate test directory; state-of-the-art classification accuracy; deployment or production inference.

## Research foundation and acknowledgments

The principal architecture reference is:

**Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).** [*ImageNet Classification with Deep Convolutional Neural Networks.*](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) Advances in Neural Information Processing Systems, 25.

This is a **collaborative project** by **Aswith Sama, Santhosh Kavi, and Divya Chintala**. According to the repository's contribution record, Aswith contributed to background research and foundational study, shared architecture development and training work, and the CPU-ready minimal working example. Contributions are documented in [`Group_contribution.txt`](Group_contribution.txt).

---

<div align="center">

**The takeaway:** Understanding a neural network begins with understanding what every layer computes — and how those computations become learnable through calculus and optimization.

**[Read the complete technical and mathematical deep dive →](DEEPDIVE.md)**

</div>
