# pytorch_mnist
learning pytorch with mnist dataset

## training

```sh
python3 train.py
```

## predict

```sh
python3 predict.py -f /data/datasets/mnist/test/0/0_1.png -m AlexNet.pkl
```

the code Calculate 1000 times to average, if you only need to calculate once, comment it out in the code.