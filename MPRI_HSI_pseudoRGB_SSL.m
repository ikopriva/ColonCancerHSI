% This code uses 3D spectral-spatial features extracted by multiscale principle
% of relevant information (MPRI) presented in the paper below:
%
%==========================================================================
% Y. Wei, S. YU, L. S. Giraldo, J. C. Principe, "Multiscale principle of  
% relevant information for hyperspectral image classification", Machine
% Learning (2023) 112:1227-1252, https://doi.org/10.1007/s10994-021-06011-9
%==========================================================================

% MPRI_HSI_pseudoRGB_SSL

clear all;close all;clc

addpath(genpath('.\MPRI_code'));

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

% Parameters for MPRI
Nclass=2; % The number of classes.
Num_labels_per_class = 500; % number of labeled pixels per class (cancer, noncancer) for SSL
                            % 500 ~ 1%

% hyperparameters of the MPRI network
DLNet.NumStages = 3;% The number of layers
DLNet.NumScale = [3 3 3];% The number of scales
DLNet.NumLDA=[1 1 1];% The number of LDA filters: NClass - 1  !!!!!!!!!
DLNet.delta=[0.3 0.3 0.3];% % kernel parameters
DLNet.beta=[2 3];   % Multi-beta

blck_tp=0; blck_tn=0; blck_fp=0; blck_fn=0;  % for estimation of micro performance

% load ground truth data
load GT_train
load GT_test

h_dataset = waitbar(0,'Progressing SSL-MPRI classification on dataset level. Please wait...');

for itr = 1:27
    waitbar(itr/27,h_dataset)

    tstart = tic; % estimate CPU time per image

    blck_tp_img=0; blck_tn_img=0; blck_fp_img=0; blck_fn_img=0;

    if dataset_flag == 1
        if itr < 18
            filename=strcat('HSI_train_',num2str(itr),'.h5');
        else
            filename=strcat('HSI_test_',num2str(itr),'.h5');
        end
        img=h5read(filename,'/img');
    elseif dataset_flag == 2
        if itr < 18
            filename=strcat('psdRGB_train_',num2str(itr),'.tiff');
        else
            filename=strcat('psdRGB_test_',num2str(itr),'.tiff');
        end
        img=double(imread(filename));
    end

    [W H B] = size(img);

    X = reshape(shiftdim(img,2),B,W*H);

    if dataset_flag == 1   % HSI data to be scaled in [0, 1] interval
        Mmax=max(max(X));
        Mmin=min(min(X));
        X = (X - Mmin)./(Mmax-Mmin);
        img=reshape(shiftdim(X,1),W,H,B);
    end

    if itr < 18
        Input_gt = GT_train(:,:,itr); 
    else
        Input_gt = GT_test(:,:,itr-17);
    end

    h_image = waitbar(0,'Progressing SSL-MPRI classification on image level. Please wait...');
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

            %select dw by dh region
            patch_img = double(img(ww_s:ww_e,hh_s:hh_e,:));
            patch_labels = double(Input_gt(ww_s:ww_e,hh_s:hh_e));

            %% This part of the code extracts MPRI-based 3D spectral-spatial feature
            if length(unique(patch_labels)) == Nclass   % skip blocks with less than Nclass labels
                TrnLabels=[]; TestLabels=[]; Tr_idx_C=[];
                Te_idx_C=[]; Te_idx_R=[]; Tr_idx_R=[];

                for i=1:Nclass
                    [R C]=find(patch_labels==(i-1));
                    Num=Num_labels_per_class;
                    idx_rand=randperm(numel(C));
                    if length(idx_rand) < Num
                        Num = length(idx_rand);
                    end
                    Tr_idx_C=[Tr_idx_C C(idx_rand(1:Num))'];
                    Tr_idx_R=[Tr_idx_R R(idx_rand(1:Num))'];
                    Te_idx_R=[Te_idx_R R(idx_rand(Num+1:end))'];
                    Te_idx_C=[Te_idx_C C(idx_rand(Num+1:end))'];
                    TrnLabels=[TrnLabels ones(1,Num)*(i-1)];
                    TestLabels=[TestLabels ones(1,numel(C)-Num)*(i-1)];
                end

                %%  Extraction of labeled MPRI features
                fprintf('\n ====== Extraction of MPRI features ======= \n')
                [f,~] = DLNet_train(patch_img,TrnLabels,Tr_idx_R,Tr_idx_C,DLNet);
                ftrain=zeros(size(f,3),numel(Tr_idx_R));
                for i=1:numel(Tr_idx_R)
                    ftrain(:,i)=reshape(f(Tr_idx_R(i),Tr_idx_C(i),:),size(f,3),1);
                end

                %% Use the rest of features as test data
                ftest=zeros(size(f,3),numel(Te_idx_R));
                for i=1:numel(Te_idx_R)
                    ftest(:,i)=reshape(f(Te_idx_R(i),Te_idx_C(i),:),size(f,3),1);
                end

                labeledX = ftrain';
                Y = TrnLabels;

                %% SSL selftrained classification
                unlabeledX = ftest';
                Mdl_SSL = fitsemiself(labeledX,Y,unlabeledX);
                testLabels_est(1,:) =  Mdl_SSL.FittedLabels';
            else  % assign the same label to all pixels in the patch
                TrnLabels=[]; TestLabels=[]; Tr_idx_C=[];  Te_idx_C=[];
                Te_idx_R=[];  Tr_idx_R=[];

                i = min(min(patch_labels));
                [R C]=find(patch_labels==i);
                Num=Num_labeled_pairs;
                idx_rand=randperm(numel(C));

                Tr_idx_C=[Tr_idx_C C(idx_rand(1:Num))'];
                Tr_idx_R=[Tr_idx_R R(idx_rand(1:Num))'];
                Te_idx_R=[Te_idx_R R(idx_rand(Num+1:end))'];
                Te_idx_C=[Te_idx_C C(idx_rand(Num+1:end))'];
                TrnLabels=[TrnLabels ones(1,Num)*i];
                TestLabels=[TestLabels ones(1,numel(C)-Num)*i];
                testLabels_est(1,:)=TestLabels;
            end

            gtest=testLabels_est;  % estimated labels
            gt = patch_labels; % true labels

            for i=1:numel(Tr_idx_R)
                gth(Tr_idx_R(i),Tr_idx_C(i))=TrnLabels(i);
            end

            for i=1:numel(Te_idx_R)
                gth(Te_idx_R(i),Te_idx_C(i))=gtest(i);
            end

            GTh_img(ww_s:ww_e,hh_s:hh_e,itr) = gth;

            TP = sum(double(and(logical(TestLabels),logical(testLabels_est))));
            TN = sum(double(~or(logical(TestLabels),logical(testLabels_est))));
            FN = sum(TestLabels) - TP;
            FP = sum(double(~logical(TestLabels))) - TN;

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

            clear patch_imag patch_labels testLabels_est f V gth
        end
    end
    close(h_image)  % end-of-the-image-loop

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
            filename_gth=strcat('GTh_MPRI_HSI_train_',num2str(itr),'_',num2str(Num_labels_per_class),'_SSL','.tiff') ;
        else
            filename_gth=strcat('GTh_MPRI_HSI_test_',num2str(itr),'_',num2str(Num_labels_per_class),'_SSL','.tiff') ;
        end
        imwrite(GTh_rgb,filename_gth,'tiff')
    elseif dataset_flag == 2
        if itr < 18
            filename_gth=strcat('GTh_MPRI_HSI_train_',num2str(itr),'_',num2str(Num_labels_per_class),'_SSL','.tiff') ;
        else
            filename_gth=strcat('GTh_MPRI_HSI_test_',num2str(itr),'_',num2str(Num_labels_per_class),'_SSL','.tiff') ;
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
        filename = strcat(' MPRI_HSI_SSL_validation_results_',num2str(Num_labels_per_class),'_labels_per_class')
    elseif dataset_flag == 2
        filename = strcat(' MPRI_RGB_SSL_validation_results_',num2str(Num_labels_per_class),'_labels_per_class')
    end
    save(filename, 'micro_sens', 'micro_spec', 'micro_bacc', 'micro_F1',...
        'micro_IoU', 'micro_ppv', 'itr', 'blck_tp', 'blck_tn', 'blck_fp', 'blck_fn',...
        'img_sens', 'img_spec', 'img_bacc', 'img_F1', 'img_IoU', 'img_ppv')

    cpu_time(itr) = toc(tstart)
end
close(h_dataset) % end-of-the-dataset-loop

