function [ r, l_fool, l_org ] = deepfool_resnet( x0, net, paras )
%[ r, l_fool, l_org ] = deepfool( x0, net, paras )
%   This is an implementation of the DeepFool algorithm on MatConvNet
%   Note that the demo programme provided by the authors is only tests with
%   a three layer LeNet and Minst digits
%
% set default parameters
class_k = 0;
norm_p = 2;
overshoot = 0.02;
max_iter = 10;
% renew the parameters
if nargin == 3
    if isfield(paras, 'class') 
        class_k = paras.class;
    end
    if isfield(paras, 'overshoot') 
        overshoot = paras.overshoot; 
    end
    if isfield(paras, 'norm_p') 
        norm_p = paras.norm_p; 
    end
elseif nargin < 3
    error('deepfool.m: not enough arguments')
elseif nargin > 3
    error('deepfool.m: too many arguments')
end
% retrieve input output names of resnet
input_cell = net.getInputs();
input_str = input_cell{1};
input_idx = net.getVarIndex(input_str);
output_cell = net.getOutputs();
output_str = output_cell{1};
output_idx = net.getVarIndex(output_str);
% initialize DeepFool
x = x0;
d = numel(x); % data dimension
size_x = size(x);
[f_out, l_org] = f(x); 
K = numel(f_out);
r = zeros(size_x);
idx_k = 1:K;
idx_k(l_org) = [];
rk_norm = inf(1, K);
fprintf('\n\t----- DeepFool iteration 1 -----\n\tcalculating derivatives for %i classes', K);
f_grad = df(x);
% main loop
for iter = 1:max_iter
    for k = idx_k
        rk_norm(k) = abs(f_out(l_org) - f_out(k))./norm(f_grad(:, l_org)-f_grad(:, k), norm_p);
    end
    [~, k_opt] = min(rk_norm);
    diff_grad = f_grad(:, l_org) - f_grad(:, k_opt);
    dr = -diff_grad * abs(f_out(l_org) - f_out(k_opt))./norm(diff_grad, norm_p).^2;
    dr = reshape(dr, size_x);
    x = x + dr;
    r = r + dr;
    [~, l_fool] = f(x0 + r*(1+overshoot)); 
    if l_fool~=l_org
        r = r*(1+overshoot);
        break;
    else
        fprintf('\n\t----- iteration %i -----\n\tcalculating derivatives for %i classes', iter+1, K);
        [f_out, ~] = f(x);
        f_grad = df(x); 
    end
end

function [out, label] = f(y)
    % forward pass to obtain the classifier output
    net.eval({input_str, y}); 
    out = net.vars(output_idx).value;
    out = out(:).';
    [~,label] = max(out);
end

function dzdx = df(y)
    dzdx = zeros(d, K);
    for kk=1:K
        outputDerValue = zeros(1, 1, K, 'single');
        outputDerValue(kk) = 1;
        % forward-backward pass
        net.eval({input_str, y}, {output_str, outputDerValue}); 
        der_input = net.vars(input_idx).der;
        dzdx(:,kk) = reshape(der_input, d, 1);
        if mod(kk,50)==1
            fprintf('\n\tclass %i-%i:\t', kk, kk+49);
        end
        fprintf('.');
    end
    fprintf('\n\t');
end
end

    