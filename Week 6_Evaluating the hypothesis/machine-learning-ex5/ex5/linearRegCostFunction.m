function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% *** Part 1.2: Regularized linear regression cost function
h = X * theta;
theta_reg = [ 0; theta(2:end) ];
% J = (1/(2*m)) * sum( (h - y).^2 ) + (lambda/(2*m)) * sum(theta_reg.^2);
J = (1/(2*m)) * (h - y)' * (h - y) + (lambda/(2*m)) * theta_reg' * theta_reg;
% *** Part 1.3: Regularized linear regression gradient
grad = (1/m) * X' * (h - y) + (lambda/m) * theta_reg;



% =========================================================================

grad = grad(:);

end
