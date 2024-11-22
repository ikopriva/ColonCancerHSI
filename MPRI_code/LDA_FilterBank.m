function V = LDA_FilterBank(InImg,TrnLabels,Tr_idx_R,Tr_idx_C,NumFilters) 
% ========= CITATION ============
% Yicong Zhou and Yantao Wei, 
% "Learning Hierarchical Spectral-Spatial Features for Hyperspectral Image Classification" IEEE Transactions on Cybernetics, 2015. 

% Yantao Wei [yantaowei@mail.ccnu.edu.cn]
% Please email me if you find bugs, or have suggestions or questions!

im=zeros(size(InImg,3),numel(Tr_idx_R));
for i=1:numel(Tr_idx_R)
    im(:,i)=InImg(Tr_idx_R(i),Tr_idx_C(i),:);
end
% options.Fisherface = 1;
options.Regu = 1;
options.ReguAlpha = 0.1;
options.ReducedDim=15;

[V, eigvalue_temp] = LDA(TrnLabels, options, im');

 %[Y, V, eigvalue]=LDA(im', TrnLabels, options.ReguAlpha);
% options=[];
% [V1, eigvalue] = PCA(im', options);
% V=[V(:,1:15) V1];

%%
