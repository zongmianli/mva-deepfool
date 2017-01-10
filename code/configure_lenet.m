%% set paths and file names
%% set working dir
workDir = '../';
%% set a pretrained model
% model = 'lenet-5-stability-training.mat';
model = 'lenet-5-baseline.mat';
fprintf(['Loading pretrained model: ' model '...\n']);
net_struct = load([workDir, 'models/', model]);
net = net_struct.net;
clear net_struct;
net = vl_simplenn_tidy(net); % add compatibility to newer versions of MatConvNet
net.layers(end) = []; % the final softmax (loss) layer should be removed in order to prevent numerical instabilities
%% select dataset and path to the file containing images
dataset = 'CIFAR-10';
fprintf(['Done. Dataset chosen: ' dataset '\n']);
imgsave = [workDir, 'results/', dataset, '_lenet/'];
loaded_struct = load([workDir, 'data/', dataset, '/test_batch.mat']);
data = loaded_struct.data;
labels = loaded_struct.labels;
data = reshape(data, [10000, 32, 32, 3]);
data = permute(data, [3, 2, 4, 1]); 
data = single(data)/255;

N = size(data, 4); % 10000 test images in CIFAR-10