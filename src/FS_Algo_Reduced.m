

addpath('Multi_libsvm/')
S_matrix = [];

%import the data from the dataset
test1 = testData;
trai1 = trainData;

TestLabel = testLabel;
TrainLabel = trainLabel;

%Get the size from the dataset

[no_row_trail,no_col_trail] = size(trai1);

ch = 1;
for i = 1 : no_row_trail; 
    for j = 1 : no_col_trail;
    Xj(ch) = trai1( i , j );
    ch = ch+1;
    end
end    

ch = 1;

%define R

R = 16 ;

%calculate laplacian score of the trail data
 
Sij = GraphConstruct_original(trai1,'KNNe',3,27) ; 
MLr = LaplacianScore(trai1,Sij) ;

%find the min and max from the laplacian 

a = min(MLr);
b = max(MLr);

%find the Normalised Laplacian 
NMLr =  (( MLr - a )/( b - a)) ;

F = trai1 ; 

[junk, index] = sort(-MLr);

%store the results of Laplacian Score
newfea = trai1(:,index);
trail_LS = trai1(:,index);
test_LS = test1(:,index);

latest_features = newfea;

%Define the set F which is equal to the laplacian feature numberss
F = index;
original_Feature = index;
ch = 1;

%Define the set Fi which is equal to the 1st laplacian feature numbers

Fi = index(1);

%cal the set differecen b/w F and Fi
[F,ix] = setdiff(F,Fi,'rows','stable');

%set S=Fi
S_matrix = Fi;
%cal number of feature in S
mod_S = size(S_matrix,1);

N = size(index,1);
row_number = 1;
NI = [];

%iterate till the number of selected features is less than total fetaures we want
while mod_S < R 
   for i=1:N
       if( ismember(F,index)  )
            for j=1:size(F,1)
                for i=1:size(S_matrix,1) 
					%caluculate the normalised mutual information 
					NI(i,:) = nmi_cn(trai1(:,F(j))',trai1(:,S_matrix(i))');
                end
				% Cal Redundancy Penalization method 
                RPI(F(j),:) = (sum( NI(:) ) )/mod_S;
				% Iteratively select the feature that maximizes Ji
                Ji(F(j),:)  =  NMLr(F(j),:) - RPI(F(j),:);
     
            end
              
        end
    end
   Ji(Ji == 0) = NaN;
   % Cal maximum value of Ji
   [max_value, index] = max( Ji(:) );
   [junk, index] = sort(-Ji);
   
   % Select the feature fi with maximum value of Ji
   Fi = index(1);
  
  % Again cal set difference of F and Fi i.e F <- F/ { fi }, S <- S?{ fi }.
   [F,ix] = setdiff(F,Fi,'rows','stable');
   S_matrix = union(S_matrix,Fi,'rows','stable') ;
   
    mod_S = size(S_matrix,1);
    N = size(index,1);
    NI =[];
    Ji = [];
    RPI= []; 
end

% Take out the important feature from Trail set using S .

train_set = trai1(:,S_matrix);
test_set  = test1(:,S_matrix);

A = setdiff(original_Feature,S_matrix,'rows','stable');
disp('S (important Features): ');

disp(S_matrix);

% Classify the feature using Knn Classification method for both Laplacian and our alogrithm.

Class = knnclassify(test_set,train_set, TrainLabel);
cp = classperf(TestLabel,Class);
fprintf('Unsupervised Feature Selection Algorithm = %g%%\n', cp.CorrectRate*100);


Class = knnclassify(test_LS(:,R),trail_LS(:,R), TrainLabel);
cp = classperf(TestLabel,Class);
fprintf('Laplacian Accuracy = %g%%\n', cp.CorrectRate*100);


gam1 =9999999999999999;
sig1 =10000016;

tt = [];
type = 'classification';

trainX = train_set;
trainY = TrainLabel;
% Classify the feature using SVM Classification method for both Laplacian and our alogrithm.

model = ovrtrain(TrainLabel, train_set, '-c 8 -g 4');
[pred ac decv] = ovrpredict(TestLabel, test_set, model);
fprintf('Unsupervised Accuracy = %g%%\n', ac * 100);


model = ovrtrain(TrainLabel, trail_LS(:,R), '-c 8 -g 4');
[pred ac decv] = ovrpredict(TestLabel, test_LS(:,R), model);
fprintf('Laplacian Accuracy = %g%%\n', ac * 100);


