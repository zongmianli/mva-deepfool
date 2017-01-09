%% set paths and file names
%% set working dir
workDir = '../';
%% set a pretrained model
model = 'imagenet-vgg-f.mat';
fprintf(['Loading pretrained model: ' model '...\n']);
net = load([workDir, 'models/', model]);
net = vl_simplenn_tidy(net); % add compatibility to newer versions of MatConvNet
net.layers(end) = []; % the final softmax (loss) layer should be removed in order to prevent numerical instabilities
%% select dataset and path to the file containing images
dataset = 'ILSVRC_2016';
fprintf(['Done. Dataset chosen: ' dataset '\n']);
imgfile = [workDir, 'data/', dataset, '/'];
imgsave = [workDir, 'results/', dataset, '/'];
%% set images names
% % whether load from another file
% imgnames = load(['imgnames_', dataset, '.mat']);
% or manually create the image names
reloadNames = true;
if reloadNames
    N_all = 8706;
    imgnames = cell(N_all, 1);
    for iter = 1:9
        imgnames{iter} = ['ILSVRC2016_test_0000000', num2str(iter)];
    end
    for iter = 10:99
        imgnames{iter} = ['ILSVRC2016_test_000000', num2str(iter)];
    end
    for iter = 100:999
        imgnames{iter} = ['ILSVRC2016_test_00000', num2str(iter)];
    end
    for iter = 1000:N_all
        imgnames{iter} = ['ILSVRC2016_test_0000', num2str(iter)];
    end
end