# Quick Start

https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

## Fashion MNIST

https://www.kaggle.com/datasets/zalando-research/fashionmnist

An MNIST-like dataset of 70,000 28x28 labeled fashion images.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. 

Each training and test example is assigned to one of the following labels:

0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot

```
# Create data loaders.
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

for i in range(3):
    for j in range(3):        
            axes[i, j].imshow(X[i*3 + j].permute(1, 2, 0))
```

[!](./assets/1.png)

