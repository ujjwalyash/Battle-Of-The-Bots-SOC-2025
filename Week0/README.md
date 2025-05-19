# Week 0: Intro to Neural Networks with PyTorch

Welcome to Week 0! This week we’ll get hands-on with neural networks using PyTorch. There's an [ipynb notebook](./Week0.ipynb) to get you started.

---

## Goals

- Understand the key components of a neural network:
  - Layers: Fully connected (Linear), Convolutional, etc.
  - Activation Functions: ReLU, Sigmoid, Tanh
  - Loss Functions: Cross-entropy, MSE
  - Optimizers: SGD, Adam, and their purpose in training
- Learn how forward and backward passes work
- Train basic models using PyTorch on MNIST or CIFAR datasets

---

## Core Concepts

### Layers

Neural networks are composed of layers. The most basic is a fully connected (Linear) layer:

```python
nn.Linear(in_features, out_features)
```

### Activation Functions

Activation functions introduce non-linearity:

```python
nn.ReLU(), nn.Sigmoid(), nn.Tanh()
```

### Loss Functions

Used to measure how far the prediction is from the truth:

```python
nn.CrossEntropyLoss(), nn.MSELoss()
```

### Optimizers

Used to update the model's weights to minimize loss:

```python
torch.optim.SGD(model.parameters(), lr=0.01)
torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## Training Loop Skeleton

```python
for epoch in range(num_epochs):
    for images, labels in dataloader:
        outputs = model(images).forward()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Challenges

1. Build a fully connected network (MLP) for MNIST digit classification.
2. Train a CNN on CIFAR-10 and visualize predictions. (bonus: Do something medical related for your resume)
3. Manually implement forward and backward passes for a small 3-layer neural network without using any layers defined in pytorch.
4. Try different activation functions and compare results.

---

## Resources

- [3Blue1Brown: What is a Neural Network? ](https://www.youtube.com/watch?v=aircAruvnKk) (The first 4 videos)  
Ignore the following links if you are able to do the assignments
- [PyTorch Beginner Tutorials](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Deep Learning with PyTorch](https://www.youtube.com/watch?v=GIsg-ZUy0MY) (Extremely long)
- [FastAI's Intro to PyTorch (Week 1)](https://course.fast.ai/Lessons/lesson1.html)

---

## TLDR

The goal of this week is to understand in depth what occurs in the forward and backward passes and how they are used together to minimize *any* kind of loss using backpropagation.  
_We will test you on this and if you score < 50% you will be kicked out of the SoC <3 /j_
