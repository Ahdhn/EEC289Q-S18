function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
for J=1:numHidden
    if J==1
        %input
        z = stack{J}.W*data; 
    else 
        %activiation
        z = stack{J}.W*hAct{J-1};
    end
    z = z + stack{J}.b;
    hAct{J}=sigmoid(z);
end
z = stack{numHidden+1}.W*hAct{numHidden};
z = z + stack{numHidden+1}.b;
E = exp(z);
pred_prob = E./sum(E,1);
hAct{numHidden+1} = pred_prob;

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end

%% compute cost
%%% YOUR CODE HERE %%%
c = log(pred_prob);
ind =sub2ind(size(c), labels', 1:size(c,2));
values = c(ind);
ceCost = -sum(values);
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
d = zeros(size(pred_prob));
d(ind)=1;
error = (pred_prob-d);

for l =numHidden+1:-1:1
    gradStack{l}.b = sum(error,2);
    if l ==1
        gradStack{l}.W = error*data';
        break;
    else
        gradStack{l}.W = error*hAct{l-1}';
    end
    error = (stack{l}.W)'*error.*hAct{l-1}.*(1-hAct{l-1});    
end
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for l = 1:numHidden+1
    wCost = wCost + 0.5*ei.lambda*sum(stack{l}.W(:).^2);
end
cost = ceCost + wCost;
for l=numHidden:-1:1
    gradStack{l}.w = gradStack{l}.W + ei.lambda*stack{l}.W;
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end