function [OutImg2, V] = DLNet_train(InImg,TrnLabels,Tr_idx_R,Tr_idx_C,DLNet)

%[OutImg2, V] = DLNet_train(InImg,TrnLabels,Tr_idx_R,Tr_idx_C,DLNet)
% 
%             Input:
%               InImg  - Hyperspectral Image Data 
%               TrnLabels   - The labels of the training samples. 
%               Tr_idx_R - Row index of the training samples. 
%               Tr_idx_C - Column index of the training samples. 
%               DLNet -   Struct value in Matlab. The fields in options
%                         that can be set:
%                            NumStages -  The number of layers
%                            NumScale -   The number of scales
%                            NumLDA -     The number of LDA filters
%                            delta -      Kernel parameters
%                            beta -       Multi-beta
%                            iteration - The number of iterations of the PRI.
%                           
%
%             Output:
%               OutImg2 - MPRI features
%               V  - LDA_Filter Bank
%

% ========= CITATION ============
% Yantao Wei [yantaowei@mail.ccnu.edu.cn]
% Please email me if you find bugs, or have suggestions or questions!

[R,C,~]=size(InImg);

NumImg = numel(Tr_idx_R);

V = cell(DLNet.NumStages,1);
ImgIdx = (1:NumImg)';

OutImg=InImg;
NumStage=DLNet.NumStages;
OutImg2=[];
if ~isfield(DLNet,'iteration')
    numI=DLNet.NumScale(1);
    DLNet.iteration = ones(numI,1)*3;
end

for i=1:NumStage

    display(['Computing PRI at layer ' num2str(i) '...'])

    OutImg = PRI_output(OutImg,DLNet.NumScale(i),DLNet.delta,DLNet.beta,DLNet.iteration);

    display(['Computing LDA filter bank and its outputs at layer ' num2str(i) '...'])
    V{i} = LDA_FilterBank(OutImg,TrnLabels,Tr_idx_R,Tr_idx_C,DLNet.NumLDA(i)); % compute LDA filter banks
    OutImg1 = LDA_output(OutImg, 2, DLNet.NumLDA(i), V{i});

    OutImg2=cat(3,OutImg2,OutImg1);

%     %% full classification map
%     OutImg=OutImg2;
%     figure
%     iptsetpref('ImshowBorder','tight')
%     iptsetpref('ImtoolInitialMagnification','fit')
%     load('indian_pines_color.mat')
% 
%     ftest=[];
%     m=1;
%     for jj=1:(size(OutImg,2))
%         for ii=1:(size(OutImg,1))
%             ftest(:,m)=reshape(OutImg(ii,jj,:),size(OutImg,3),1);
%             m=m+1;
%         end
%     end
% 
%     ftrain=[];
% 
%     m=1;
%     for jj=1:numel(TrnLabels)
%         ftrain(:,m)=reshape(OutImg(Tr_idx_R(jj),Tr_idx_C(jj),:),size(OutImg,3),1);
%         m=m+1;
%     end
% 
%     TestLabels=ones(1,R*C);
% 
%     [label_index_expected,~,~] = KNN(1,ftrain',TrnLabels',ftest',TestLabels');
%     colormap(1,:)=[];
%     OutImg=[];
%    % imshow(reshape(label_index_expected,R,C),colormap)
%     %%
     OutImg=OutImg1;
end
end









