# Machine Learning

## Supervised vs Unsupervised Learning

Machine Learning trains models to make predictions or find structure in data without being explicitly programmed with rules. Supervised Learning uses labeled Training Data — paired examples of inputs and correct outputs — to learn a mapping that Generalizes to new observations. Regression predicts continuous outcomes; Classification predicts discrete categories. The Learning Algorithm adjusts Model Training parameters to minimize a Loss Function that measures how wrong predictions are.

Unsupervised Learning finds structure in data without labels. Clustering groups similar observations together based on feature similarity. The absence of a Target Variable means there is no definitive right answer — evaluation is inherently harder, requiring domain judgment about whether discovered structure is meaningful.

The Training Process iterates over data, computing the Cost Function and adjusting parameters via Optimization. Gradient Descent is the foundational optimization algorithm: compute the gradient of the loss with respect to parameters, then step in the direction that decreases the loss. The Learning Rate controls step size — too large and the algorithm overshoots; too small and Convergence is slow. The algorithm seeks a Global Minimum of the loss landscape, though it may settle at a Local Minimum.

## Neural Networks

Neural Networks approximate complex functions through layers of simple computations. An Artificial Neuron computes a weighted sum of its inputs plus a Bias, then passes the result through an Activation Function that introduces nonlinearity. The Sigmoid Function squashes output to (0, 1); the ReLU Function outputs zero for negative inputs and the input itself for positive inputs. Without nonlinear activation functions, stacking layers produces no more expressive power than a single linear transformation.

The Network Architecture is defined by the number and size of layers. The Input Layer receives raw features; Hidden Layers transform them into increasingly abstract representations; the Output Layer produces predictions. Weights — the parameters of each connection — are learned during training. Forward Propagation computes predictions layer by layer from input to output. Backpropagation computes gradients of the loss with respect to all Weights by applying the chain rule in reverse through the network.

Deep Learning uses networks with many Hidden Layers. Each layer learns a level of abstraction: early layers detect simple patterns; later layers combine them into complex representations. The Vanishing Gradient problem occurs when gradients shrink exponentially as they propagate back through many layers, stalling learning in early layers. ReLU activations partially address this.

## Training in Practice

Epochs count how many complete passes through the Training Data the algorithm makes. Mini-batch training uses a Batch Size of examples at each gradient step rather than the full dataset (too slow) or single examples (Stochastic Gradient, too noisy). PyTorch Library implements this workflow via Tensors, Autograd for Automatic Differentiation, and a Neural Network Module that tracks parameters. The Training Loop iterates over batches, computes loss, calls backward propagation, and applies an Optimizer like SGD Optimizer or Adam Optimizer to update parameters.

Evaluating a neural network requires the same Train Test Split discipline as any other model. Model Evaluation PyTorch measures loss and accuracy on held-out data. Overfitting is especially common in deep networks because they have millions of parameters and can memorize Training Data exactly. Regularization techniques — dropout, weight decay, data augmentation — prevent this. Model Interpretability and Explainable AI tools like SHAP Values help diagnose what a trained network has learned, which is essential for detecting bias and building trust.
