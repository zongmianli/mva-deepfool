% to setup: install matconvnet-1.0-beta23 and download the imagenet-vgg-f
% pretrained model
clear
%% load pretrained model
net = load('imagenet-vgg-f.mat');
net = vl_simplenn_tidy(net); % add compatibility to newer versions of MatConvNet
net.layers(end) = []; % the final softmax (loss) layer should be removed in order to prevent numerical instabilities
%% image loading, preprocessing to fit the model
im = imread('data/images/000005.jpg');
im_ = single(im);
im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
im_ = im_ - net.meta.normalization.averageImage;
%% fool the convnet
paras.class_k = 0; % set k non-zero to make the program reach a specific class
paras.norm_p = 2; % choose the p value for p-norm
paras.overshoot = 0.02; % make sure we always reach a adverserial pertubation
[ r, l_fool, l_org ] = deepfool(im_, net, paras);
%% visualization
figure;
subplot(1,3,1); 
imagesc(im_/256); 
title(['Original image labeled as class ', num2str(l_org)]);
subplot(1,3,2); 
imagesc((im_+r)/256); 
title(['Perturbed image labeled as class ', num2str(l_fool)]);
subplot(1,3,3); 
imagesc(r); 
title('Perturbation [scaled]');