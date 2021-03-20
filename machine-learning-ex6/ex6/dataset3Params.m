function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%a
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

pos_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];


cost_diff = zeros(size(pos_vals)(1,2),  size(pos_vals)(1,2))
counter = 0
for sigma_index=1:size(pos_vals)(1,2)
  for c_index=1:size(pos_vals)(1,2)
    counter = counter + 1;
    my_predictions = svmTrain(X, y, pos_vals(c_index), @(x1, x2) gaussianKernel(x1, x2, pos_vals(sigma_index)));
    best_predictions = svmPredict(my_predictions, Xval);
    cost_diff(sigma_index, c_index) = mean(double(best_predictions ~= yval));
  endfor
endfor
cost_diff
%min = cost_diff(1,1)
%sigma = pos_vals(1)
%C = pos_vals(1)
%for sigma_index=1:size(pos_vals)(1,2)
%  for c_index=1:size(pos_vals)(1,2)
%     if cost_diff(sigma_index,c_index)<min
%       min = cost_diff(1,1);
%       sigma = pos_vals(sigma_index);
%       C = pos_vals(c_index);
%     endif
%  endfor
%endfor
[minval, best_sigma_index] = min(min(cost_diff,[],2));
[minval, best_c_index] = min(min(cost_diff,[],1));
sigma = pos_vals(best_sigma_index);
C = pos_vals(best_c_index);
%ValCost = mean(double(predictions ~= yval)) / (2 * size(Xval, 2));


% =========================================================================

end
