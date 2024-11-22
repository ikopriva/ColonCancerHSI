function [img_tensorSSA] = TensorSSA(u,L,patch_train)

% Tensor SSA
[W,H,B]=size(patch_train);

w=2*u+1; w2=w*w;  % research region

patch_train = padarray(patch_train,[u,u],'symmetric','both');
Id=zeros(L,W*H);
Fea_cube=zeros(L,W*H,B);

%adaptive embedding
k=0;
for i=1:W
    for j=1:H
        i1=i+u;j1=j+u;k=k+1;
        testcube=patch_train(i1-u:i1+u,j1-u:j1+u,:);
        m=reshape(testcube,[w2,B]);

        %NED
        center=m((w2+1)/2,:);NED=zeros(1,w2);
        for ii=1:w2
            NED(:,ii)=sqrt(sum(power((m(ii,:)/norm(m(ii,:))-center/norm(center)),2)));%NED
        end
        [~,ind]=sort(NED);
        index=ind(1:L);
        Id(:,k)=index;
        Fea_cube(:,k,:)=m(index,:);
    end
end

%T-SVD decomposition
[U,S,V] = tSVD(Fea_cube,1); %rank=1,i.e.,Low-rank representation
clear Fea_cube;
C = tProdact(U,S);
VT=tTranspose(V);
Feacube_proc=tProdact(C,VT);

%Reprojection
New_pad_img=zeros(W+w-1,H+w-1,B);
repeat=zeros(W+w-1,H+w-1);
kk=0;
for i=1:W
    for j=1:H
        kk=kk+1;
        rec_m=zeros(w2,B);
        rec_m(Id(:,kk),:)=Feacube_proc(:,kk,:);
        dd=reshape(rec_m,[w,w,B]);

        rec_col=zeros(w2,1);
        rec_col(Id(:,kk))=1;

        i1=i+u;j1=j+u;
        New_pad_img(i1-u:i1+u,j1-u:j1+u,:)=New_pad_img(i1-u:i1+u,j1-u:j1+u,:)+dd;
        repeat(i1-u:i1+u,j1-u:j1+u)=repeat(i1-u:i1+u,j1-u:j1+u)+reshape(rec_col,w,w);
    end
end

ind = (repeat==0);
repeat(ind)=1;

New_pad_img=New_pad_img./repeat;
img_tensorSSA=New_pad_img(u+1:W+u,u+1:H+u,:);

end