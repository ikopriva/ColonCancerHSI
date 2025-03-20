%==========================================================================
%
% Grassmann_HSI
%
% (c) Ivica Kopriva, March 2025
%==========================================================================

% Grassmann_HSI

clear all; close all; clc

addpath(genpath('.\TensorSSA_code\'));
addpath(genpath('.\TensorSSA_code\tcSVD-master'));

dataset_flag = 1;     % 1 - HSI data original

p = 5;              % patch size: (2p+1) x (2p+1) 
                     % p=0 ===> no patch

Num_labeled_pairs = 600; % number of labels per category used to construct Grassmann points

Nclass = 2;   % Number of classess                    

dimSubspace = 20;   % subspace dimensions of Grassmannian points                    

pixel_flag = 1;     % 1 - pixel-to-subspace distance


pseudo_labels_flag = 0; % 0 - use true labels
                        % 1 - use pseudo labels

mean_flag = 0;          % 0 - selected pixel is used
                        % 1 - mean value of the patch around pixel is used

numwav=351;            % 450nm to 800 nm;  


blck_tp=0; blck_tn=0; blck_fp=0; blck_fn=0;  % for estimation of micro performance

% load ground truth data
load GT_train
load GT_test

h_dataset = waitbar(0,'Progressing Grassman-TensorSSA classification on dataset level. Please wait...');

for itr = 1:27
    waitbar(itr/27,h_dataset)

    tstart = tic; % estimate CPU time per image

    blck_tp_img=0; blck_tn_img=0; blck_fp_img=0; blck_fn_img=0;

    u=5; L=60; % patch-emebedding combination

    if itr < 18
        filename=strcat('HSI_train_',num2str(itr),'.h5');
    else
        filename=strcat('HSI_test_',num2str(itr),'.h5');
    end
    img=h5read(filename,'/img');

    [H W B] = size(img)

    X = reshape(shiftdim(img,2),B,W*H);

    if itr < 18
        Input_gt = GT_train(:,:,itr); 
    else
        Input_gt = GT_test(:,:,itr-17);
    end

    h_image = waitbar(0,'Progressing Grassman-TensorSSA classification on image level. Please wait...');
    % Grassman semi-suervised classifier works on the patch basis of the size 230x258
    dh=230; dw=258;
    WI=floor(W/dw)*dw; HI=floor(H/dh)*dh;

    for hh= 1:dh:HI
        hh_s = hh;
        if hh_s <= HI-dh
            hh_e = hh_s + (dh-1);
        else
            hh_e = HI;
        end

        for ww=1:dw:WI
            ww_s = ww;
            if ww <= WI-dw
                ww_e = ww_s + (dw-1);
            else
                ww_e = WI;
            end

            %select dx by dw region
            patch_img = double(img(hh_s:hh_e,ww_s:ww_e,:));
            patch_labels = double(Input_gt(hh_s:hh_e,ww_s:ww_e));
            nc = sum(sum(patch_labels)); nnc=sum(sum(not(patch_labels)));

            %%
            if (length(unique(patch_labels)) == Nclass) && (Num_labeled_pairs <= nc) && (Num_labeled_pairs <= nnc)   % skip blocks with less than Nclass labels
                TrnLabels=[];
                TestLabels=[];
                Tr_idx_C=[];
                Te_idx_C=[];
                Te_idx_R=[];
                Tr_idx_R=[];

                display('Training image:')
                itr
                display('Block row start:')
                hh_s
                display('Block column start:')
                ww_s

                %% Construct representatives Y0 and Y1
                for i=1:Nclass
                    [R C]=find(patch_labels==(i-1));
                    Num=Num_labeled_pairs;
                    idx_rand=randperm(numel(C));
                    Tr_idx_C=[Tr_idx_C C(idx_rand(1:Num))'];
                    Tr_idx_R=[Tr_idx_R R(idx_rand(1:Num))'];
                    Te_idx_R=[Te_idx_R R(idx_rand(Num+1:end))'];
                    Te_idx_C=[Te_idx_C C(idx_rand(Num+1:end))'];
                    TrnLabels=[TrnLabels ones(1,Num)*(i-1)];
                    TestLabels=[TestLabels ones(1,numel(C)-Num)*(i-1)];
                end

                % Construct TensorSSA features
                u = 5; L = 60;
                patch_img = TensorSSA(u,L,patch_img);

                X0=zeros(B,Num);
                X1=zeros(B,Num);
                if mean_flag == 0 % use selected pixel
                    for i=1:Num
                        X0(:,i)=patch_img(Tr_idx_R(i),Tr_idx_C(i),:);
                        X1(:,i)=patch_img(Tr_idx_R(i+Num),Tr_idx_C(i+Num),:);
                    end
                elseif mean_flag == 1 % use mean patch value around selected pixel
                    for i=1:Num
                        % Noncancer class
                        blk = patch_extractor(patch_img,Tr_idx_R(i),Tr_idx_C(i),p);
                        X0(:,i) = squeeze(mean(mean(blk,1),2));

                        % Cancer class
                        blk = patch_extractor(patch_img,Tr_idx_R(i+Num),Tr_idx_C(i+Num),p);
                        X1(:,i) = squeeze(mean(mean(blk,1),2));
                    end
                end

                % Estimate orthonormal bases: Grassmann points
                XX = [X0 X1];
                labels = [ones(1,Num_labeled_pairs) 2*ones(1,Num_labeled_pairs)];
                [affinity_x, B_x, begB_x, enddB_x, mu_X]  = average_affinity(XX,labels,dimSubspace);

                % Testing
                for i=1:numel(Te_idx_R)
                    if mean_flag == 0
                        X_out(:,i)=patch_img(Te_idx_R(i),Te_idx_C(i),:);
                    elseif mean_flag == 1  % assign mean value
                        blk = patch_extractor(patch_img,Te_idx_R(i),Te_idx_C(i),p);
                        X_out(:,i) = squeeze(mean(mean(blk,1),2));
                    end
                end

                for el=1:2
                    X_outm = X_out - mu_X(:,el);    % make data zero mean for distance calculation
                    BB=B_x(:,begB_x(el):enddB_x(el));
                    Xproj = (BB*BB')*X_outm;
                    Dproj = X_outm - Xproj;
                    D(el,:) = sqrt(sum(Dproj.^2,1));
                end
                [~, testLabels_est] = min(D);
                testLabels_est = testLabels_est - 1;
                clear D X_out
            else  % assign the same label to all pixels in the block
                display('ONE LABEL ONLY !!!!!!!!!!')

                TrnLabels=[];
                TestLabels=[];
                Tr_idx_C=[];
                Te_idx_C=[];
                Te_idx_R=[];
                Tr_idx_R=[];

                display('Training image:')
                itr
                display('Block row start:')
                hh_s
                display('Block column start:')
                ww_s

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

            gtest=testLabels_est;
            gt = patch_labels;

            for i=1:numel(Tr_idx_R)
                gth(Tr_idx_R(i),Tr_idx_C(i))=TrnLabels(i);
            end

            for i=1:numel(Te_idx_R)
                gth(Te_idx_R(i),Te_idx_C(i))=gtest(i);
            end

            GTh_img_train(hh_s:hh_e,ww_s:ww_e,itr) = gth;
            GT_img(hh_s:hh_e,ww_s:ww_e) = gt;

            TP = sum(double(and(logical(TestLabels),logical(testLabels_est))));
            TN = sum(double(~or(logical(TestLabels),logical(testLabels_est))));
            FN = sum(TestLabels) - TP;
            FP = sum(double(~logical(TestLabels))) - TN;

            % image based performance
            blck_tp_img = blck_tp_img + TP;
            blck_tn_img = blck_tn_img + TN;
            blck_fp_img = blck_fp_img + FP;
            blck_fn_img = blck_fn_img + FN;
            F1_train_img = 2*blck_tp_img/(2*blck_tp_img + blck_fp_img + blck_fn_img)

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

            clear patch_imag patch_labels testLabels_est gth gt
        end
    end
    close(h_image) % end-of-image loop

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

    if itr < 18
        filename_gth=strcat('GTh_TensorSSA_HSI_train_',num2str(itr),'_',num2str(Num_labels_per_class),'_Grassmann','.tiff') ;
    else
        filename_gth=strcat('GTh_TensorSSA_HSI_test_',num2str(itr),'_',num2str(Num_labels_per_class),'_Grassmann','.tiff') ;
    end
    imwrite(GTh_rgb,filename_gth,'tiff')

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


    filename = strcat(' TensorSSA_HSI_Grassmann_validation_results_',num2str(Num_labels_per_class),'_labels_per_class')

    save(filename, 'micro_sens', 'micro_spec', 'micro_bacc', 'micro_F1',...
        'micro_IoU', 'micro_ppv', 'itr', 'blck_tp', 'blck_tn', 'blck_fp', 'blck_fn',...
        'img_sens', 'img_spec', 'img_bacc', 'img_F1', 'img_IoU', 'img_ppv')

    cpu_time(itr) = toc(tstart)
end
close(h_dataset) % end-of-dataset-loop
