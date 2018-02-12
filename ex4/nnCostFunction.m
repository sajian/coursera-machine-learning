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
% size(Theta1) = [25 401]
% size(Theta2) = [10 26]
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1); % number of examples

% Preallocate
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

% Part 1, Feedforward-------------------------------------------

% Recode y
% e.g. if y = 5, recode it to a vector [0 0 0 0 1 0 0 0 0 0]
y_recode = zeros(num_labels, m);
label_pos = y' + num_labels*(0:m-1); % Calculate the linear index of the position where should be labeled as 1
y_recode(label_pos) = 1;
y = y_recode';

% Input layer
a_1 = [ones(m, 1) X]; % size(X) = [5000 401]

% Hidden layer
a_layer2 = sigmoid(a_1 * Theta1');
a_layer2 = [ones(m, 1) a_layer2]; % size(a_layer2) = [5000 26]

% Output layer
hypothesis = sigmoid(a_layer2 * Theta2'); % size(hypothesis) = [5000 10]

% Calculate the cost
% The inner sum for adding up K labels' cost of each example.
% The outer sum for adding up every examples' cost.
J = (1/m) * sum(sum(-y .* log(hypothesis) - (1-y) .* log(1 - hypothesis), 2))...
    + (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));

% Part 2, Backpropagation---------------------------------------

for t = 1:m
    % Step 1, Feedforward pass
    a_1 = [1; X(t, :)'];                % size(a_1) = [401 1];
    
    z_2 = Theta1 * a_1;
    a_2 = [1; sigmoid(z_2)];            % size(a_2) = [26 1];
    
    a_3 = sigmoid(Theta2 * a_2);        % size(a_3) = [10 1];

    % Step 2, Delta in output layer
    delta_3 = a_3 - y(t, :)';
    
    % Step 3, Delta in hidden layer
    delta_2 = Theta2' * delta_3 .* [1; sigmoidGradient(z_2)];
    delta_2 = delta_2(2:end);
    
    % Step 4, Accumulate the gradient
    Theta1_grad = Theta1_grad + delta_2 * a_1';
    Theta2_grad = Theta2_grad + delta_3 * a_2';
end

% Part 3, Regularized-------------------------------------------

% Step 5, Obtain the gradients
Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
