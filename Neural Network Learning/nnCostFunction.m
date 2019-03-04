function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%   NNCOSTFUNCTION Implements the neural network cost function for a two layer
%   neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta_1 and Theta_2, the weight matrices
% for our 2 layer neural network
Theta_1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta_2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta_1_grad = zeros(size(Theta_1));
Theta_2_grad = zeros(size(Theta_2));


% Part One: Feedforward / Cost Function

% K is the number of classes.
K = num_labels;
Y = eye(K)(y, :);

a_1 = [ones(m, 1), X];

z_2 = a_1 * Theta_1';
a_2 = sigmoid(z_2);
a_2 = [ones(size(a_2, 1), 1), a_2];

z_3 = a_2 * Theta_2';
a_3 = sigmoid(z_3);

cost = sum((-Y .* log(a_3)) - ((1 - Y) .* log(1 - a_3)), 2);
J = (1 / m) * sum(cost);

% Regularized Cost Function
Theta1_without_bias = Theta_1(:, 2:end);
Theta2_without_bias = Theta_2(:, 2:end);

% Use sum squared function
reg  = (lambda / (2 * m)) * (sum(sumsq(Theta1_without_bias)) + sum(sumsq(Theta2_without_bias)));
J += reg;

% Part Two: Vectorized Backpropagation
d_3 = a_3 - Y;                                             
d_2 = (d_3 * Theta_2) .* [ones(size(z_2,1),1) sigmoidGradient(z_2)];   

D_1 = d_2(:,2:end)' * a_1;    
D_2 = d_3' * a_2;    

Theta_1_grad_unregularized = (1/m) * D_1;
Theta_2_grad_unregularized = (1/m) * D_2;

% Adding Regularization to the Gradients
for i = 2:size(Theta_1_grad_unregularized, 2)
  Theta_1_grad_unregularized(:, [i]) = Theta_1_grad_unregularized(:, [i]) + ((lambda / m) * Theta_1(:, [i]));
endfor

for i = 2:size(Theta_2_grad_unregularized, 2)
  Theta_2_grad_unregularized(:, [i]) = Theta_2_grad_unregularized(:, [i]) + ((lambda / m) * Theta_2(:, [i]));
endfor

Theta_1_grad = Theta_1_grad_unregularized;
Theta_2_grad = Theta_2_grad_unregularized;


% Unroll gradients
grad = [Theta_1_grad(:) ; Theta_2_grad(:)];


end

