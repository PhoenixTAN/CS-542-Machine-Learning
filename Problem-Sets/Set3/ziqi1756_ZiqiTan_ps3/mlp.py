import numpy as np
import matplotlib.pyplot as plt


class TwoLayerMLP(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax - cross entropy loss
                                z1     a1                      
    The outputs of the second fully-connected layer are the scores for each class.
    """

    
    def __init__(self, input_size, hidden_size, output_size, std=1e-4, activation='relu'):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.output_size = output_size
        self.params = {}    # define a parameters dictionary

        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.activation = activation

    ####################################################################################
    
    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        _, C = W2.shape   # W2: H*C
        N, D = X.shape    # X: N*D

        # Compute the forward pass
        ##########################################################################
        # write your own code where you see [PLEASE IMPLEMENT]
        
        # Perform the forward pass, computing the class scores for the input.
        # Store the result in the scores variable, which should be an array of
        # shape (N, C).
        ##########################################################################
        z1 = np.dot(X, W1) + b1  # 1st layer activation, N*H
        # print("b1")
        # print(b1.shape)
        # print("z1")
        # print(z1)
        hidden = None                # activation function  N*H

        # 1st layer nonlinearity, N*H
        if self.activation is 'relu':
            # [PLEASE IMPLEMENT]
            # hidden = np.log(1+np.exp(z1)) # This does not perform well.
            hidden = (z1 + np.abs(z1))/2
            # raise NotImplementedError('ReLU forward not implemented')
        elif self.activation is 'sigmoid':
            # [PLEASE IMPLEMENT]
            hidden = 1/(1+np.exp(-z1))
            # raise NotImplementedError('Sigmoid forward not implemented')
        else:
            raise ValueError('Unknown activation type')

        # print("hidden")
        # print(hidden)

        # [PLEASE IMPLEMENT] 2nd layer activation, N*C
        # hint: involves W2, b2
        # W2: Second layer weights; has shape (H, C)
        # b2: Second layer biases; has shape (C,)
        scores = np.dot(hidden, W2) + b2      
        # print("scores")
        # print(scores)
        
        ###########################################################################
        #                            END OF YOUR CODE
        ###########################################################################
        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # cross-entropy loss with log-sum-exp
        A = np.max(scores, axis=1) # N*1

        F = np.exp(scores - A.reshape(N, 1))  # N*C
        
        # P is the softmax
        P = F / np.sum(F, axis=1).reshape(N, 1)  # N*C
        # print("Softmax = ")
        # print(P)
       
        # print("scores.T = ")
        # print(scores.T)
        # print("y = ")
        # print(y)
        # print("choose = ")
        # print(-np.choose(y, scores.T))
        loss = np.mean(-np.choose(y, scores.T) + np.log(np.sum(F, axis=1)) + A)
        # triky way to avoid overflow
        
        # add regularization terms
        loss += 0.5 * reg * np.sum(W1 * W1)
        loss += 0.5 * reg * np.sum(W2 * W2)
        # print("loss: ", loss)

        # Backward pass: compute gradients
        grads = {}
        ###########################################################################
        # write your own code where you see [PLEASE IMPLEMENT]
        #
        # Compute the backward pass, computing the derivatives of the weights
        # and biases. Store the results in the grads dictionary. For example,
        # grads['W1'] should store the gradient on W1, and be a matrix of same size.
        # You should define the hidden variable in the part where you implement the 
        # different activation functions. "hidden" is the output after you apply 
        # the activation function for the hidden layer. 
        # Hint: you should apply different activation functions on "z1" and get "hidden".   
        ###########################################################################
        
        # architecture: 
        # input --(w1, b1)-- fully connected layer --get "z1"--- 
        # ReLU ---get "hidden"--- fully connected layer ---get "scores"--- softmax ---(y)--- cross-entropy loss
        
        # output layer
        # loss = (P - true_label)
        # P = softmax(scores)
        # dloss/dscores = dloss/dP * dP/dscores 
        # = (P - true_label) / P(1-P) * P(1-P) = P - true_label
        
        Y = self.trans_y_to_matrix(y)
        
        dscore = P - Y # [PLEASE IMPLEMENT] partial derivative of loss wrt. the logits (dL/dz)
        # print("dscore=", dscore)
        dW2 = np.dot(hidden.T, dscore)/N      # partial derivative of loss wrt. W2  H*C
        db2 = np.mean(dscore, axis=0)         # partial derivation of loss wrt. b2
        
        # hidden layer
        dhidden = np.dot(dscore, W2.T)   # N*H
        dz1 = None
        if self.activation is 'relu':
            # [PLEASE IMPLEMENT]
            # dz1 = np.multiply(dhidden, 1+np.exp(-z1))
            dz1 = dhidden * (z1 > 0)    # .astype('int32')
            # raise NotImplementedError('ReLU backward not implemented')
        elif self.activation is 'sigmoid':
            dz1 = np.multiply(dhidden, np.multiply(hidden, 1-hidden))
            # raise NotImplementedError('Sigmoid backward not implemented')
        else:
            raise ValueError('Unknown activation type')

        # first layer
        dW1 = np.dot(X.T, dz1)/N  # [PLEASE IMPLEMENT]  
        db1 = np.mean(dz1, axis=0)  # [PLEASE IMPLEMENT]
        ###########################################################################
        #                            END OF YOUR CODE
        ###########################################################################

        grads['W2'] = dW2 + reg*W2
        grads['b2'] = db2
        grads['W1'] = dW1 + reg*W1
        grads['b1'] = db1
        return loss, grads

    #################################################################################

    
    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_epochs=10,
            batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
        Stochastic refers to a randomly determined process.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(int(num_train / batch_size), 1)
        epoch_num = 0

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        grad_magnitude_history = []
        train_acc_history = []
        val_acc_history = []

        np.random.seed(1)
        for epoch in range(num_epochs):
            # fixed permutation (within this epoch) of training data
            perm = np.random.permutation(num_train)

            # go through minibatches
            for it in range(iterations_per_epoch):
                X_batch = None
                y_batch = None

                # Create a random minibatch
                idx = perm[it*batch_size:(it+1)*batch_size]
                X_batch = X[idx, :]
                y_batch = y[idx]

                # Compute loss and gradients using the current minibatch
                loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
                loss_history.append(loss)

                # do gradient descent
                for param in self.params:
                    self.params[param] -= grads[param] * learning_rate

                # record gradient magnitude (Frobenius) for W1
                grad_magnitude_history.append(np.linalg.norm(grads['W1']))

            # Every epoch, check train and val accuracy and decay learning rate.
            # Check accuracy
            train_acc = (self.predict(X_batch) == y_batch).mean()
            val_acc = (self.predict(X_val) == y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            if verbose:
                print('Epoch %d: loss %f, train_acc %f, val_acc %f'%(
                    epoch+1, loss, train_acc, val_acc))

            # Decay learning rate
            learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'grad_magnitude_history': grad_magnitude_history, 
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }


    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """

        ###########################################################################
        # [PLEASE IMPLEMENT]
        # hint: it should be very easy
        # just implement the architecture
        z1 = np.dot(X, self.params['W1']) + self.params['b1']
        
        # 1st layer nonlinearity, N*H
        if self.activation is 'relu':
            # [PLEASE IMPLEMENT]
            # hidden = np.log(1+np.exp(z1)) # This does not perform well.
            hidden = (z1 + np.abs(z1))/2
            # raise NotImplementedError('ReLU forward not implemented')
        elif self.activation is 'sigmoid':
            # [PLEASE IMPLEMENT]
            hidden = 1/(1+np.exp(-z1))
            # raise NotImplementedError('Sigmoid forward not implemented')
        else:
            raise ValueError('Unknown activation type')

        scores = np.dot(hidden, self.params['W2']) + self.params['b2']
        
        y_pred = np.argmax(scores, axis=1)    # Returns the indices of the maximum values along an axis.
        ###########################################################################

        return y_pred

    def trans_y_to_matrix(self, y):
        # y is a 1*N
        Y = np.zeros((len(y), self.output_size))
        for i in range(len(y)):
            Y[i][y[i]] = 1
        
        return Y    # N*C
