function [m_affinity, B_x, begB_x, endB_x, mu_X] = average_affinity(X,labels,dimSubspace)
%
% This routine computes average affinity for a given dataset and its current partition
%
% Inputs:
% X - dxn array of data (d: number of features; n - number of data)
% labels - vector of (pseud)labels (from 1 toc nc) for n-data produced by
%          some clustering algorithm
% dimSubspace - subspace dimension for a particular dataset
%
% Outputs:
% m_affinity: estimated averaged affinity for the given tition (the smaller the more incoherent are subspaces)
%
% B_x - estimated bases of nc subspaces
% begB_x - beining indexes of each basis
% endB_x - ending indexes of each basis
      
nc=max(labels);

% Allocate space
B_x = zeros(size(X,1), nc * dimSubspace);
begB_x = zeros(1, nc); endB_x = zeros(1, nc);
mu_X = zeros(size(X,1),nc);

for c=1:nc % Through all categories
    Xc = X(:, labels == c); % Samples of the chosen category
    X_c = normc(Xc);        % Normalize data...
    mu_c=mean(X_c,2);
    X_c = Xc - mu_c; %mean(X_c,2);  % ... and make it zero mean
    mu_X(:,c) = mu_c; 
    dx_c = min(dimSubspace, size(Xc,2)); % Subspace dimension, but not less than the actual size
    [T,S,V] = svds(X_c, dx_c); % Singular Value Decomposition with reduction of dimensions to dx_c
    
    if c==1 % First...
        begB_x(c) = 1;         % ... beginning and ...
        endB_x(c) = size(T,2); % ... ending index.
    else
        begB_x(c) = endB_x(c-1)+1; % Beginning index is determined by previous ending index
        endB_x(c) = endB_x(c-1)+size(T,2); % Ending index is a cummulative sum of the previous ones
    end
    B_x(:, begB_x(c):endB_x(c)) = T; % Copy corresponding vectors to the cummulative result
end
B_x(:,endB_x(nc)+1 :end) = []; % Cut out the surplus of the allocated space

% Calculate principal angles between subspaces     
aff_x=0;  cnt=0;
for i=1:nc-1
  UX_i  =B_x( :,begB_x(i) :endB_x(i)  );                 
  
  for j=i+1:nc
      cnt=cnt+1;
      UX_j  =B_x( :,begB_x(j) :endB_x(j)  ); 
      angles = cos(mPrinAngles(UX_i,UX_j)); 
      angles = angles.*angles;
      dx_m=min(size(UX_i,2),size(UX_j,2)); 
      aff_x = aff_x + sqrt(sum(angles)/dx_m); 
  end
end

% Affinities of minimal principal angles between the subspaces 
m_affinity  = aff_x /cnt;
end