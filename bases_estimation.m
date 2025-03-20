function [B_x, begB_x, endB_x, mu_X] = bases_estimation(X,labels,dimSubspace)
%
% This routine computes bases for given partitions of dataset X
%
% Inputs:
% X - dxn array of data (d: number of features; n - number of data)
% labels - vector of (pseud)labels (from 1 toc nc) for n-data produced by
%          some clustering algorithm
% dimSubspace - dimension of the subspaces

%
% Outputs:
%
% B_x - estimated bases of nc subspaces
% begB_x - beining indexes of each basis
% endB_x - ending indexes of each basis
% mu_X - mean valued of each particular subspace
      
nc=max(labels);

% Allocate space
B_x = zeros(size(X,1), nc * dimSubspace);
begB_x = zeros(1, nc); endB_x = zeros(1, nc);
mu_X = zeros(size(X,1),nc);

for c=1:nc % Through all categories
    Xc = X(:, labels == c); % Samples of the chosen category
    X_c = normc(Xc);        % Normalize data...
    mu_c=mean(X_c,2);
    X_c = Xc - mu_c;  % ... and make it zero mean
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
end