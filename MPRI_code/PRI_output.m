function OutImg= PRI_output(InImg,NumFilters,delta,beta,iteration)

[Im1,Im2,Im3]=size(InImg);

 fil=[3 7 11]; % patch sizes

iter_n_plus_1=zeros(Im1,Im2,Im3);

OutImg=zeros(Im1, Im2, Im3*NumFilters*numel(beta));
iter_n_plus_BS=zeros(Im1,Im2,NumFilters*Im3);

for kkk=1:numel(beta)

for i=1:NumFilters
    Para1=2*delta(i)^2;
    Para2=1000000;
    S=fil(i);
    C_x=zeros(S*S,S*S);
    C_x_s=zeros(S*S,S*S);
    InImg1=padarray(InImg,[floor(S/2) floor(S/2)],'symmetric','both');
    iter_n=InImg1;
    hhs=zeros(S,S,2);
    for si=1:S
        for sj=1:S
            hhs(si,sj,:)=[si sj];
        end
    end
    
   hhx=(hhs/(S^2*Para2));
%    tic 
for ii=1:iteration(i)
   if ii>1
    InImg2=padarray(iter_n_plus_1,[floor(S/2) floor(S/2)],'symmetric','both');
    iter_n=InImg2;
    clear InImg2;
   end
for j=1:size(InImg1,1)-S+1
       
        for k=1:size(InImg1,2)-S+1
            
            temp0=InImg1(j:j+S-1,k:k+S-1,:);
            temp0=cat(3,temp0,hhx);
            temp1=zeros(Im3+2,S*S);
            temp2=reshape(temp0(round(S/2),round(S/2),:),Im3+2,1);
            
            
            tempn0=iter_n(j:j+S-1,k:k+S-1,:);
            tempn0=cat(3,tempn0,hhx);
            
            tempn1=zeros(Im3+2,S*S);
            tempn2=reshape(tempn0(round(S/2),round(S/2),:),Im3+2,1);
            
            sc=zeros(1,S^2);
            scn=zeros(1,S^2);
%             t=1;
            
            
            temp1 = ToVector(temp0);
            tempn1 = ToVector(tempn0);
            sc=-sum((ones(S*S,1)*tempn2'-temp1).^2,2)';
            scn=-sum((ones(S*S,1)*tempn2'-tempn1).^2,2)';
            
            
            xx0=temp1*temp1';
            xxnd=tempn1*tempn1';
            xxxxx0d=diag(xx0);
            xxxxxnd=diag(xxnd);
            
            C_x=(ones(S*S,1)*xxxxxnd')'-2*xxnd+ones(S*S,1)*xxxxxnd';
            
            C_x_s=(ones(S*S,1)*xxxxxnd')'+ones(S*S,1)*xxxxx0d'-2*tempn1*temp1';
            
                    
            
            sc=exp(sc/Para1);
            scn=exp(scn/Para1);
            c=sum(exp(-C_x_s(:)/Para1))/sum(exp(-C_x(:)/Para1));
             
           iter_n_plus_2(j,k,:)=c*(1-beta(kkk))/(beta(kkk))*(sum(ones(Im3+2,1)*scn.*tempn1',2))./(sum(sc))+(sum(ones(Im3+2,1)*sc.*temp1',2))/(sum(sc))-c*(1-beta(kkk))/beta(kkk)*(sum(scn))*tempn2/(sum(sc));
        
        end
end
iter_n_plus_1=iter_n_plus_2(:,:,1:end-2);
end
iter_n_plus_BS(:,:,(i-1)*Im3+1:i*Im3)=iter_n_plus_1;
end
%   toc  
 OutImg(:,:,(kkk-1)*Im3*NumFilters+1:kkk*Im3*NumFilters)=iter_n_plus_BS;
end

