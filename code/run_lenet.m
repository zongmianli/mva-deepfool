% to setup: install matconvnet-1.0-beta23
% clear
%%
configure_lenet
% N: number of images in the dataset. N=10000 test images in CIFAR-10
% imgfile: path to the file containing images
% imgnames: N_all x 1 cell array containing the image names
%%
idx_img = 1:(N-1); % set index of images we want to process
see_imgs = false; % visualize results
save_img = false; % save images
if ~exist('r_norm', 'var')
    r_norm = zeros(1, N);
    img_norm = zeros(1, N);
end
for i_img = idx_img
    fprintf('\nProcessing Image %i (%i in total)', i_img, N);
    % fool the convnet
    paras.class_k = 0; % set k non-zero to make the program reach a specific class
    paras.norm_p = 2; % choose the p value for p-norm
    paras.overshoot = 0.02; % make sure we always reach a adverserial pertubation
    im = data(:,:,:,i_img);
    im_nor = data(:, :, :, i_img:(i_img+1));
    im_nor(:, :, :, 2) = im_nor(:, :, :, 1);
    [ r, l_fool, l_org ] = deepfool_lenet(im_nor, net, paras);
    rs(:, i_img) = r(:);
    imgs(:, i_img) = im_nor(:);
    r_norm(i_img) = norm(r(:), 2);
    img_norm(i_img) = norm(im_nor(:), 2);
    % visualization
    if see_imgs
        figure(i_img);
        subplot(1,3,1);
        imagesc(im);
        title(['Original label: ', num2str(l_org)]);
        subplot(1,3,2);
        imagesc(im+r);
        title(['Perturbed label: ', num2str(l_fool)]);
        subplot(1,3,3);
        imagesc(r*255);
        title('Perturbation [scaled]');
        savefig([imgsave, 'fool_cifar_test_', num2str(i_img), '_vis.fig']);
        saveas(gcf, [imgsave, 'fool_cifar_test_', num2str(i_img), '_vis.jpeg']);
    end
    % save data
    if save_img
        imwrite(im+r, [imgsave, 'fool_cifar_test_', num2str(i_img), '.jpeg']);
    end
%     if 0 %mod(i_img, 200)==1
%         save([imgsave, dataset, '.mat'], 'rs', 'r_norm', 'imgs', 'img_norm');
%     end
end
save([imgsave, dataset, '.mat'], 'rs', 'r_norm', 'imgs', 'img_norm');