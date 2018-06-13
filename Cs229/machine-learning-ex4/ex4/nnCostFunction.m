function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
[m,n] = size(X);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m,1), X]; %(m,n+1)
z2 = a1*Theta1'; %(m,d1)
a2 = sigmoid(z2); %(m, d1)
a2 = [ones(size(a2,1),1), a2]; %(m, d1+1)
z3 = a2*Theta2'; %(m, c)
a3 = sigmoid(z3); %(m, c)
y = eye(num_labels)(y,:);
J = 1/m*sum(sum(-y.*log(a3)-(1-y).*log(1-a3)));
J = J + lambda/2/m*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

%课程中省略了a3-y的由来，可以发现下面dz3化简可得
da3 = (a3-y)./((1-a3).*a3);
dz3 = da3.*sigmoidGradient(z3); %化简可得dz3=a3-y
Theta2_grad = dz3'*a2;
da2 = dz3*Theta2;
dz2 = da2(:,2:end).*sigmoidGradient(z2);
Theta1_grad = dz2'*a1;

reg_1 = lambda/m*Theta1;  
reg_2 = lambda/m*Theta2;  
reg_1(:,1) = zeros(size(reg_1,1),1);  
reg_2(:,1) = zeros(size(reg_2,1),1); 
Theta1_grad = Theta1_grad/m+reg_1;
Theta2_grad = Theta2_grad/m+reg_2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
