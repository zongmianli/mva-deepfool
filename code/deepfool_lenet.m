function [ r, l_fool, l_org ] = deepfool_lenet( x0, net, paras )
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
% initialize DeepFool
x = x0;
d = numel(x)/2; % data dimension, divided by 2 since there are two identical inputs
size_x = size(x);
[f_out, l_org] = f(x); 
K = numel(f_out);
r = zeros(size_x);
idx_k = 1:K;
idx_k(l_org) = [];
rk_norm = inf(1, K);
% fprintf('\n\t----- DeepFool iteration 1 -----\n\tcalculating derivatives for %i classes', K);
f_grad = df(x);
% main loop
for iter = 1:max_iter
    for k = idx_k
        rk_norm(k) = abs(f_out(l_org) - f_out(k))./norm(f_grad(:, l_org)-f_grad(:, k), norm_p);
    end
    [~, k_opt] = min(rk_norm);
    diff_grad = f_grad(:, l_org) - f_grad(:, k_opt);
    dr = -diff_grad * abs(f_out(l_org) - f_out(k_opt))./norm(diff_grad, norm_p).^2;
    dr = reshape(dr, size_x(1:3));
    x(:,:,:,1) = x(:,:,:,1) + dr;
    x(:,:,:,2) = x(:,:,:,1);
    r(:,:,:,1) = r(:,:,:,1) + dr;
    r(:,:,:,2) = r(:,:,:,1);
    [~, l_fool] = f(x0 + r*(1+overshoot)); 
    if l_fool~=l_org
        r = r*(1+overshoot);
        r = r(:,:,:,1);
        break;
    else
%         fprintf('\n\t----- iteration %i -----\n\tcalculating derivatives for %i classes', iter+1, K);
        [f_out, ~] = f(x);
        f_grad = df(x); 
    end
end

function [out, label] = f(y)
    % forward pass to obtain the classifier output
    res = vl_simplenn(net, y, [], [], 'Mode', 'test');
    out = res(end).x(:).';
    out = out(1:10); % discard identical outputs
    [~,label] = max(out);
end

function dzdx = df(y)
    dzdx = zeros(d, K);
    res = vl_simplenn(net, y, [], [], 'Mode', 'test'); % forward pass
%     fprintf('\n\tclass 1-10:\t');
    for kk=1:K
       %% creat a tensor so that its inner product with the network outputs is the kk-th output
        dzdy = zeros(1, 1, K, 2, 'single');
        dzdy(1, 1, kk, 1) = 1;
       %% backward pass
        % each kk loop calculates the partial derivative df_k/dx_i (k=1,..,K, i=1,...,N)
        res_d = vl_simplenn(net, y, dzdy, res, 'skipForward', true, 'Mode', 'test');
       %% dzdx(:,kk) records the derivative of the kk-th output w.r.t. the inputs
        dzdx_temp = res_d(1).dzdx;
        dzdx_temp = dzdx_temp(:, :, :, 1);
        dzdx(:,kk) = reshape(dzdx_temp(:), d, 1);
%         fprintf('.');
    end
%     fprintf('\n\t');
end
end

    