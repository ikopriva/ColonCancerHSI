function OutImg = LDA_output(InImg, NumBeta, NumFilters, V)
% ========= CITATION ============

% Yantao Wei [yantaowei@mail.ccnu.edu.cn]
% Please email me if you find bugs, or have suggestions or questions!

cnt = 0;
num=1;

[ImgX, ImgY, NumChls] = size(InImg);

im=zeros(NumChls,ImgX*ImgY);
for j=1:ImgY
    for i=1:ImgX
        temp=InImg(i,j,:);
        im(:,num)=reshape(temp,numel(temp),1);
        num=num+1;
    end
end

for j = 1:NumFilters
    cnt = cnt + 1;
    OutImg(:,:,cnt) = reshape(V(:,j)'*im,ImgX,ImgY);  % convolution output
end