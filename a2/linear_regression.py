import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import get_data_loaders
import matplotlib as plt
import pandas as pd


class LinearRegressionModel(nn.Module):
    """LinearRegressionModel is the linear regression regressor.
    This class handles only the standard linear regression task.
    :param num_param: The number of parameters that need to be initialized.
    :type num_param: int
    """

    def __init__(self, num_param):
        ## TODO 1: Set up network
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.linear(num_param, 1) #what is number of outputs? 1
        #pass
        

    def forward(self, x):
        """forward generates the predictions for the input
        
        This function does not have to be called explicitly. We can do the
        following 
        
        .. highlight:: python
        .. code-block:: python
            model = LinearRegressionModel(1, mse_loss)
            predictions = model(X)
    
        :param x: Input array of shape (n_samples, n_features) which we want to
            evaluate on
        :type x: typing.Union[np.ndarray, torch.Tensor]
        :return: The predictions on x
        :rtype: torch.Tensor
        """
        ## TODO 2: Implement the linear regression on sample x
        out = self.linear(x)
        return out
        #pass


def data_transform(sample):
    ## TODO: Define a transform on a given (x, y) sample. This can be used, for example
    ## for changing the feature representation of your data so that Linear regression works
    ## better.
    x, y = sample
    return sample  ## You might want to change this


def mse_loss(output, target):
    """Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`output` and target :math:`target`.
    
    The loss can be described as:
    .. math::
        \\ell(x, y) = L = \\operatorname{mean}(\\{l_1,\\dots,l_N\\}^\\top), \\quad
        l_n = \\left( x_n - y_n \\right)^2,
    where :math:`N` is the batch size. 
    :math:`output` and :math:`target` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.
    
    :param output: The output of the model or our predictions
    :type output: torch.Tensor
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor]
    :return: torch.Tensor
    :rtype: torch.Tensor
    """
    ## TODO 3: Implement Mean-Squared Error loss. 
    # Use PyTorch operations to return a PyTorch tensor
    #target.sub(output)
    # for i, x in enumerate(output.shape[]):
    #     for i, x in enumerate(output.numpy()):
            
    #     target.sub(output)


    loss = nn.mse_loss()
    return loss(output, target) # ... :>? 
    #pass


def mae_loss(output, target):
    """Creates a criterion that measures the mean absolute error (l1 loss)
    between each element in the input :math:`output` and target :math:`target`.
    
    The loss can be described as:
    .. math::
        \\ell(x, y) = L = \\operatorname{mean}(\\{l_1,\\dots,l_N\\}^\\top), \\quad
        l_n = \\left| x_n - y_n \\right|,
    where :math:`N` is the batch size. 
    :math:`output` and :math:`target` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.
    
    :param output: The output of the model or our predictions
    :type output: torch.Tensor
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor]
    :return: torch.Tensor
    :rtype: torch.Tensor
    """
    ## TODO 4: Implement L1 loss. Use PyTorch operations.
    # Use PyTorch operations to return a PyTorch tensor.
    loss = torch.nn.l1
    return loss(output, target) # ... :< 

    #pass


if __name__ == "__main__":
    ## Here you will want to create the relevant dataloaders for the csv files for which 
    ## you think you should use Linear Regression. The syntax for doing this is something like:
    # Eg:
    train_loader, val_loader, test_loader = get_data_loaders(DS1.csv, #WHAT DATA TO TRAIN ON 
                        transform_fn=data_transform  # Can also pass in None here
    )
                        # train_val_test=[TRAIN/VAL/TEST SPLIT], 
                        # batch_size=YOUR BATCH SIZE)

    ## Now you will want to initialise your Linear Regression model, using something like
    # Eg:
    model = LinearRegressionModel(2)

    ## Then, you will want to define your optimizer (the thing that updates your model weights)
    # Eg:
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    ## Now, you can start your training loop:
    # Eg:
    model.train()
    #define total time steps
    TOTAL_TIME_STEPS = 1 #increase if runs 
    for t in range(TOTAL_TIME_STEPS):
      for batch_index, (input_t, y) in enumerate(train_loader):
        optimizer.zero_grad()
    
        preds = model(input)
          #Feed the input to the model
    
        loss = loss_fn(preds, y)  # You might have to change the shape of things here.
        
        loss.backward() 
        optimizer.step()
        
    ## Don't worry about loss.backward() for now. Think of it as calculating gradients.

    ## And voila, your model is trained. Now, use something similar to run your model on
    ## the validation and test data loaders:
    # Eg: 
    model.eval()
    #elias: do twice for val and test
    for batch_index, (input_t, y) in enumerate(test_loader):
    
      preds = model(input)
    
      loss = loss_fn(preds, y) 


    for batch_index, (input_t, y) in enumerate(val_loader):
    
      preds = model(input)
    
      loss = loss_fn(preds, y)

    #
    ## You don't need to do loss.backward() or optimizer.step() here since you are no
    ## longer training.

    #pass

def plot_data2(w1, w2, b):
    data2 = pd.read_csv('data/DS2.csv')
    data2.columns = ['x', 'y']
    plt.scatter(data2['x'],data2['y']) 
    x = np.arange(-6, 6)
    plt.plot(x, w1*x*x*x + w2*x +b, linestyle='-', c = 'r')

    plt.title('Features2')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def plot_data1(w1, w2, b):
    data1 = pd.read_csv('data/DS1.csv')
    data1.columns = ['x1','x2','y']
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    ax.scatter(data1['x1'],data1['x2'],data1['y'])
    x = np.linspace(-60,60)
    y = np.linspace(-50,90)

    X,Y = np.meshgrid(x,y)
    Z=w1*X + w2*Y + b
    surf = ax.plot_surface(X, Y, Z)

    ax.set_title('The Features')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')


    plt.show()
