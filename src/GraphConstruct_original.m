function S = GraphConstruct_original(x,model,K,K2)
%% ===========Parameter============
%Construct Graph
%date: 2010-12-1
%author:Yazhou Ren www.scut.edu.cn  email:yazhou.ren@mail.scut.edu.cn 
%Input
%  x    ---  data,N*R
%  model---  model of Similarity matrix. 
%  K    ---  neighbor size (for KNN,KFN)
%  K2   ---  farest neighbor size (if 'KFNN' only)
%Output
%  S    ---  Similarity(Weight) matrix  or cell

%% ============main============
N = size(x,1);
S=zeros(N,N);% similarity matrix
switch model
    case 'KNN1'        
        distance = Mydistance_original(x);     
        [~,index] = sort(distance,'ascend');
        for i=1:N
            for j=1:1+K
                S(i,index(j,i)) = 1;
                S(index(j,i),i) = 1;
            end
        end      
        S = S-diag(diag(S));
     case 'KNNe'        
        distance = (Mydistance_original(x)).^2;     %20140318%% d(xi,xj)^2
        width=mean(mean(distance));        %t = the mean of all the elements
        sim=exp(-distance/width);          % exp(-(d(xi,xj)^2/t))
        [~,index] = sort(distance,'ascend');
        for i=1:N
            for j=1:1+K
                S(i,index(j,i)) = sim(i,index(j,i));
                S(index(j,i),i) = S(i,index(j,i));
            end
        end   
        S = S-diag(diag(S));
     case 'KFN1'  % k farest nodes      
        distance = Mydistance_original(x);     
        [~,index] = sort(distance,'descend');
        for i=1:N
            for j=1:K %attention
                S(i,index(j,i)) = 1;
                S(index(j,i),i) = 1;
            end
        end             
     case 'KFNe'  % k farest nodes      
        distance = (Mydistance_original(x)).^2;     %20140318
        width=mean(mean(distance));        %t = the mean of all the elements
        sim=exp(-distance/width);          % exp(-(d(xi,xj)^2/t))
        [~,index] = sort(distance,'descend');
        for i=1:N
            for j=1:K %attention
                S(i,index(j,i)) = sim(i,index(j,i));
                S(index(j,i),i) = S(i,index(j,i));
            end
        end      
    case 'KFNN1' % KNN+KFN (Sn & Sf)
        S = cell(1,2);
        Sn = zeros(N,N);
        Sf = zeros(N,N);
        distance = Mydistance_original(x);     
        % KNN Sn
        [~,index] = sort(distance,'ascend');
        for i=1:N
            for j=1:1+K
                Sn(i,index(j,i)) = 1;
                Sn(index(j,i),i) = 1;
            end
        end       
        Sn = Sn-diag(diag(Sn));     %
        % KFN Sf
%         K2=N-K-1;
        [~,index] = sort(distance,'descend');
        for i=1:N
            for j=1:K2 %attention
                Sf(i,index(j,i)) = 1;
                Sf(index(j,i),i) = 1;
            end
        end          
        S{1} = Sn;
        S{2} = Sf;
    case 'KFNNe' % KNN+KFN (Sn & Sf)
        S = cell(1,2);
        Sn = zeros(N,N);
        Sf = zeros(N,N);
        distance = (Mydistance_original(x)).^2;     %20140318
        width=mean(mean(distance));
        sim=exp(-distance/width);
        % KNN Sn
        [~,index] = sort(distance,'ascend');
        for i=1:N
            for j=1:1+K
                Sn(i,index(j,i)) = sim(i,index(j,i));
                Sn(index(j,i),i) = Sn(i,index(j,i));
            end
        end         
        Sn = Sn-diag(diag(Sn));
        % KFN Sf
%         K2=N-K-1;
        [~,index] = sort(distance,'descend');
        for i=1:N
            for j=1:K2 %attention
                Sf(i,index(j,i)) = sim(i,index(j,i));
                Sf(index(j,i),i) = Sf(i,index(j,i));
            end
        end          
        S{1} = Sn;
        S{2} = Sf;        
end
