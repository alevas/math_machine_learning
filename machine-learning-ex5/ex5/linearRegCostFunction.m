function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad
0
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

sum_j_sq  = 0;
for j = 1:m 
  h = X(j,:) * theta;
  sum_j_sq = sum_j_sq + (h - y(j))^2;
end
J = sum_j_sq/(2 * m);

theta_sum = 0;
for j=2:size(theta)
  theta_sum = theta_sum + theta(j)^2;
endfor

J = J + lambda / (2 * m) * theta_sum;

%%%%%%%%%1.3%%%%%%%%%
sum_j = 0;
for i=1:size(theta)
  for j = 1:m 
    h = X(j,:) * theta;
    sum_j = (h - y(j)) * X(j,i);
    grad(i) = grad(i) + sum_j/m;
  end
end


for j = 2:size(theta)
  mini_factor = lambda/m*theta(j);
  grad(j) = grad(j) + mini_factor;
end






% =========================================================================

grad = grad(:);

end
