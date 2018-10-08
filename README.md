# MNIST digits classification using neural network
First, clone or download the repository. Then enter the path.
This neural network uses two kinds of activate function (ReLU and sigmoid). You can run the Python files.
```python
python mnist_neural_network_relu.py
```
or
```python
python mnist_neural_network_sigmoid.py
```
The Python function load_data() is used to load train set and test set.
```python
X_train, y_train, X_test, y_test = load_data()
```
Then use the function run() to start training network.
```python
run(X_train, y_train, X_test, y_test)
```

