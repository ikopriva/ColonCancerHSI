function [kappa, acc, acc_O, acc_A]=MPRI(Input,Input_gt,DLNet,NClass,iter,Per)


kappa=zeros(iter,1);
acc=zeros(NClass,iter);
acc_O=zeros(iter,1);
acc_A=zeros(iter,1);
for Iter=1:iter

TrnLabels=[];
TestLabels=[];
Tr_idx_C=[];
Te_idx_C=[];
Te_idx_R=[];
Tr_idx_R=[];
for i=1:NClass   
    [R C]=find(Input_gt==i);
%    Num=Trnum(i);
    Num=ceil(numel(C)*Per);
    idx_rand=randperm(numel(C));
    Tr_idx_C=[Tr_idx_C C(idx_rand(1:Num))'];
    Tr_idx_R=[Tr_idx_R R(idx_rand(1:Num))'];
    Te_idx_R=[Te_idx_R R(idx_rand(Num+1:end))'];
    Te_idx_C=[Te_idx_C C(idx_rand(Num+1:end))'];
    TrnLabels=[TrnLabels ones(1,Num)*i];
    TestLabels=[TestLabels ones(1,numel(C)-Num)*i]; 
end
% ===========================================================
%% Training
fprintf('\n ======Training ======= \n')
[f,~] = DLNet_train(Input,TrnLabels,Tr_idx_R,Tr_idx_C,DLNet); % 

ftrain=zeros(size(f,3),numel(Tr_idx_R));
for i=1:numel(Tr_idx_R)
    ftrain(:,i)=reshape(f(Tr_idx_R(i),Tr_idx_C(i),:),size(f,3),1);
end

clear TrnData_ImgCell; 

%% Testing 
fprintf('\n ====== Testing ======= \n')

ftest=zeros(size(f,3),numel(Te_idx_R));
for i=1:numel(Te_idx_R)
    ftest(:,i)=reshape(f(Te_idx_R(i),Te_idx_C(i),:),size(f,3),1);
end

label_index_expected=zeros(iter,numel(Te_idx_R));
[label_index_expected(Iter,:),~,~] = KNN(1,ftrain',TrnLabels',ftest',TestLabels');
%% Evaluate_results

[kappa(Iter), acc(:,Iter), acc_O(Iter), acc_A(Iter)] = evaluate_results(label_index_expected(Iter,:), TestLabels)
end