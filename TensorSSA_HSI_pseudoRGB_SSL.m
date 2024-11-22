% This code uses 3D spectral-spatial features extracted by Tensor Spectrum 
% Singularity Analysis (TensorSSA) presented in the paper below:
%==========================================================================
% H. Fu, G. Sun, A. Zhang, B. Shao, J. Ren, and X. Jia, "Tensor Singular 
% Spectrum Amalysis for 3-D Feature Extraction in Hyperspectral Images,"
% IEEE Transctions of Geoscience and Remote Sensing, vol. 61, 
% article no. 5403914, 2023.
%==========================================================================

% TensorSSA_HSI_pseudoRGB_SSL

clear all;close all;clc

addpath(genpath('.\TensorSSA_code\'));
addpath(genpath('.\TensorSSA_code\tcSVD-master'));
%addpath(genpath('.\TensorSSA\functions'));
%addpath(genpath('.TensorSSA\libsvm-3.18'));

% Select either hyperspectral (HSI) images or co-registered color (RGB)
% images:

dataset_flag = 2;     % 1 - HSI data original
                      % 2 - pseudoRGB data

%% load dataset

% It is assumed that both hyperspectral data and RGB data are located in
% the local directory. They are available for download from the link below:
% https:\\....

% Data are organized in train partition (17 images from 9 patients) and test 
% partition (10 images from 5 patients). That is necessary for training and
% testing deep networks. For semi-supervised learning that work on
% patch-by-patch basis the partitions are treated equally, i.e., there are
% no train and test partitions.


%% Start

Num_class=2; % The number of classes.
Num_labels_per_class = 500; % number of labeled pixels per class (cancer, noncancer) for SSL
                            % 500 ~ 1%

blck_tp=0; blck_tn=0; blck_fp=0; blck_fn=0;  % for estimation of micro performance

% load ground truth data
load GT_train
load GT_test

h_dataset = waitbar(0,'Progressing SSL-TensorSSA classification on dataset level. Please wait...');

for itr = 1:27
    waitbar(itr/27,h_dataset)

    tstart = tic; % estimate CPU time per image

    blck_tp_img=0; blck_tn_img=0; blck_fp_img=0; blck_fn_img=0;

    if dataset_flag == 1
        u=5; L=60; % patch-emebedding combination

        if itr < 18
            filename=strcat('HSI_train_',num2str(itr),'.h5');
        else
            filename=strcat('HSI_test_',num2str(itr),'.h5');
        end
        img=h5read(filename,'/img');
    elseif dataset_flag == 2
        u = 5; L = 8; % patch-emebedding combination

        if itr < 18
            filename=strcat('psdRGB_train_',num2str(itr),'.tiff');
        else
            filename=strcat('psdRGB_test_',num2str(itr),'.tiff');
        end
        img=double(imread(filename));
        img = img/255; % scale to [0, 1] range
    end

    [W H B] = size(img);

    X = reshape(shiftdim(img,2),B,W*H);

    if itr < 18
        Input_gt = GT_train(:,:,itr) + 1;  % to take into account class numbering from 1
    else
        Input_gt = GT_test(:,:,itr-17) + 1; % to take into account class numbering from 1
    end

    h_image = waitbar(0,'Progressing SSL-TensorSSA classification on image level. Please wait...');
    % SSL works on the patch basis of the size 230x258
    dw=230; dh=258;
    WI=floor(W/dw)*dw; HI=floor(H/dh)*dh;
       
    i_patch=0; 
    for ww= 1:dw:WI
        ww_s = ww;
        if ww <= WI-dw
            ww_e = ww_s + (dw-1);
        else
            ww_e = WI;
        end

        for hh=1:dh:HI
            i_patch = i_patch + 1;
            waitbar(i_patch/24,h_image)
            hh_s = hh;
            if hh <= HI-dh
                hh_e = hh_s + (dh-1);
            else
                hh_e = HI;
            end

            %select dw x dh region
            patch_img = double(img(ww_s:ww_e,hh_s:hh_e,:));
            patch_labels = double(Input_gt(ww_s:ww_e,hh_s:hh_e));

            [W_p H_p B] = size(patch_img);

            %% Tensor SSA
            patch_img_tensorSSA = TensorSSA(u,L,patch_img); % 3D spectral-spatial TensorSSA features

            % SSL based classification
            Labels=reshape(patch_labels,W_p*H_p,1);
            Vectors=reshape(patch_img_tensorSSA,W_p*H_p,B);

            trainVectors=[];trainLabels=[];train_index=[];
            testVectors=[]; testLabels=[]; test_index = [];
            rng('default');

            for k=1:1:Num_class
                index=find(Labels==k);
                if length(index) > Num_labels_per_class
                    perclass_num=length(index);
                    Sam_img = Num_labels_per_class/perclass_num;
                    Vectors_perclass=Vectors(index,:);
                    c=randperm(perclass_num);
                    select_train=Vectors_perclass(c(1:ceil(perclass_num*Sam_img)),:);
                    train_index_k=index(c(1:ceil(perclass_num*Sam_img)));
                    select_train=Vectors_perclass(1:ceil(perclass_num*Sam_img),:);
                    train_index_k=index(1:ceil(perclass_num*Sam_img));
                    train_index=[train_index;train_index_k];;
                    trainVectors=[trainVectors;select_train];
                    trainLabels=[trainLabels;repmat(k-1,ceil(perclass_num*Sam_img),1)];

                    select_test=Vectors_perclass(c(ceil(perclass_num*Sam_img)+1:perclass_num),:);
                    test_index_k=index(c(ceil(perclass_num*Sam_img)+1:perclass_num));
                    select_test=Vectors_perclass(ceil(perclass_num*Sam_img)+1:perclass_num,:);
                    test_index_k=index(ceil(perclass_num*Sam_img)+1:perclass_num);
                    test_index=[test_index;test_index_k];
                    testVectors=[testVectors;select_test];
                    testLabels=[testLabels;repmat(k-1,perclass_num-ceil(perclass_num*Sam_img),1)];
                end
            end
            % true labels with feature vectors for SSL classifier
            labeledX = trainVectors;
            Y = trainLabels;

            % SSL based classification
            unlabeledX = testVectors;

            Mdl_SSL = fitsemiself(labeledX,Y,unlabeledX);
            testLabels_est(1,:) =  Mdl_SSL.FittedLabels';

            Labels_est = zeros(W_p*H_p,1);
            Labels_est(train_index,1)=trainLabels;
            Labels_est(test_index,1)=testLabels_est;
            gth = reshape(Labels_est,W_p,H_p);
            gt = patch_labels;
            gtest=testLabels_est;

            GTh_img(ww_s:ww_e,hh_s:hh_e,itr) = gth;

            TP = sum(double(and(logical(testLabels'),logical(testLabels_est))));
            TN = sum(double(~or(logical(testLabels'),logical(testLabels_est))));
            FN = sum(testLabels) - TP;
            FP = sum(double(~logical(testLabels))) - TN;

            % image based performance
            blck_tp_img = blck_tp_img + TP;
            blck_tn_img = blck_tn_img + TN;
            blck_fp_img = blck_fp_img + FP;
            blck_fn_img = blck_fn_img + FN;

            % micro performance
            blck_tp = blck_tp + TP;
            blck_tn = blck_tn + TN;
            blck_fp = blck_fp + FP;
            blck_fn = blck_fn + FN;

            figure(100)
            subplot(1,2,1)
            imagesc(gt)
            title('GT')
            axis('square')
            subplot(1,2,2)
            imagesc(gth)
            title('GTH')
            axis('square')
            pause(3)
            clear patch_imag patch_labels testLabels_est TestLabels gth
        end
    end
    close(h_image) % end-of-image-loop

    cmap=[0 0 1; 1 1 0];  %blue:: noncancer;  yellow: cancer;

    if itr < 18
        GT_int=uint8(GT_train(:,:,itr));
    else
        GT_int=uint8(GT_test(:,:,itr-17));
    end

    GT_rgb = ind2rgb(GT_int,cmap);
    GTh_img_int = uint8(GTh_img(:,:,itr));
    GTh_rgb = ind2rgb(GTh_img_int,cmap);

    figure(itr)
    subplot(1,2,1)
    imagesc(GT_rgb)
    title('GT')
    axis('square')
    set(gca,'Xtick',[]);
    set(gca,'Ytick',[]);
    subplot(1,2,2)
    imagesc(GTh_rgb)
    title('Estimated GT')
    axis('square')
    set(gca,'Xtick',[]);
    set(gca,'Ytick',[]);

    if dataset_flag == 1
        if itr < 18
            filename_gth=strcat('GTh_TensorSSA_HSI_train_',num2str(itr),'_',num2str(Num_labels_per_class),'_SSL','.tiff') ;
        else
            filename_gth=strcat('GTh_TensorSSA_HSI_test_',num2str(itr),'_',num2str(Num_labels_per_class),'_SSL','.tiff') ;
        end
        imwrite(GTh_rgb,filename_gth,'tiff')
    elseif dataset_flag == 2
        if itr < 18
            filename_gth=strcat('GTh_TensorSSA_HSI_train_',num2str(itr),'_',num2str(Num_labels_per_class),'_SSL','.tiff') ;
        else
            filename_gth=strcat('GTh_TensorSSA_HSI_test_',num2str(itr),'_',num2str(Num_labels_per_class),'_SSL','.tiff') ;
        end
        imwrite(GTh_rgb,filename_gth,'tiff')
    end

    display('Image based performance metrics:')
    % image based performance
    img_sens(itr) = blck_tp_img/(blck_tp_img + blck_fn_img)
    img_spec(itr) = blck_tn_img/(blck_tn_img + blck_fp_img)
    img_bacc(itr) = (img_sens(itr) + img_spec(itr))/2
    img_F1(itr) = 2*blck_tp_img/(2*blck_tp_img + blck_fp_img + blck_fn_img)
    img_IoU(itr) = img_F1(itr)/(2-img_F1(itr))
    img_ppv(itr) = blck_tp_img/(blck_tp_img + blck_fp_img)

    display('Micro performance metrics:')
    % micro performance
    micro_sens = blck_tp/(blck_tp + blck_fn)
    micro_spec = blck_tn/(blck_tn + blck_fp)
    micro_bacc = (micro_sens + micro_spec)/2
    micro_F1 = 2*blck_tp/(2*blck_tp + blck_fp + blck_fn)
    micro_IoU = micro_F1/(2-micro_F1)
    micro_ppv = blck_tp/(blck_tp + blck_fp)

    if dataset_flag == 1
        filename = strcat(' TensorSSA_HSI_SSL_validation_results_',num2str(Num_labels_per_class),'_labels_per_class')
    elseif dataset_flag == 2
        filename = strcat(' TensorSSA_RGB_SSL_validation_results_',num2str(Num_labels_per_class),'_labels_per_class')
    end
    save(filename, 'micro_sens', 'micro_spec', 'micro_bacc', 'micro_F1',...
        'micro_IoU', 'micro_ppv', 'itr', 'blck_tp', 'blck_tn', 'blck_fp', 'blck_fn',...
        'img_sens', 'img_spec', 'img_bacc', 'img_F1', 'img_IoU', 'img_ppv')

    cpu_time(itr) = toc(tstart)
end
close(h_dataset) % end-of-dataset-loop