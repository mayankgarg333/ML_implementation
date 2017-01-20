
%%
% load the wine data and divide in test and training set

close all;
clear all;
wine = importdata('wine.data');
%get the class label for each sample
winelabel = wine(:,1);
%each row of winefeature matrix contains the 13 features of each sample
winefeature = wine(:,2:end);
%generate training and test data
train = []; %stores the training samples
test = [];  %stores the test samples
for i=1:3
    ind{i} = find(winelabel==i);
    len = length(ind{i});
    t = randperm(len);
    half = round(len/2);
    train = [train; wine(ind{i}(t(1:half)), :)];
    test = [test; wine(ind{i}(t(half+1:end)), :)];
end


%% Linear classification
% storing the training data
X_train=train(:,2:end);
T_train=train(:,1);
T=zeros(length(T_train),3);  
% getting the T matrix in the [1 0 0] format as explained in the report
% 1-k type coding scheme classification
for i=1:length(X_train)
    T(i,T_train(i))=1;
end
% calculating X-tilde
X_train=[ones(length(X_train),1) X_train];
% calculating the W using equation 4.16 Bishop
W=inv(X_train'*X_train)*X_train'*T;
% predicting Y using W
Y_predict_train=X_train*W;
class_Y_train=zeros(length(T_train),3);
for i=1:length(X_train)
   [max_value, index] = max(Y_predict_train(i,:));
    class_Y_train(i,index)=1;
    class_Y_train_V(i)=index;
end
z1=find((T_train==class_Y_train_V')==0);
figure(1)
% plotting the classification of the training data
scatter3(Y_predict_train(:,1),Y_predict_train(:,2),Y_predict_train(:,3),T_train*40,T,'LineWidth',3)
hold on
scatter3(Y_predict_train(z1,1),Y_predict_train(z1,2),Y_predict_train(z1,3),500*ones(length(z1),1),class_Y_train(z1,:),'+','LineWidth',3)
title('Training set prediction, Linear classifier')

% Test data
X_test=test(:,2:end);
X_test=[ones(length(X_test),1) X_test];
Y_predict_test=X_test*W;
class_Y_Test=zeros(length(X_test),3);
for i=1:length(X_test)
   [max_value, index] = max(Y_predict_test(i,:));
    class_Y_Test(i,index)=1;
    class_Y_Test_V(i)=index;
end
T_test=test(:,1);
T2=zeros(length(T_test),3);
for i=1:length(X_test)
    T2(i,T_test(i))=1;
end
% plotting of the test data set... 
figure(2)
scatter3(Y_predict_test(:,1),Y_predict_test(:,2),Y_predict_test(:,3),T_test*40,T2,'LineWidth',3)
% only 2 predition was incorrect, rest was correct...
hold on
z2=find((T_test==class_Y_Test_V')==0);
scatter3(Y_predict_test(z2,1),Y_predict_test(z2,2),Y_predict_test(z2,3),500*ones(length(z2),1),class_Y_Test(z2,:),'+','LineWidth',3)
title('Test set prediction, Linear classifier')



%% Fisher discriminant

%calculate means mk as given in equation 4.42
% Also the X matrix for a particular class
X1=train(find(train(:,1)==1),2:end);
m1=sum(X1)'/length(find(train(:,1)==1));
X1=X1';
X2=train(find(train(:,1)==2),2:end);
m2=sum(X2)'/length(find(train(:,1)==2));
X2=X2';
X3=train(find(train(:,1)==3),2:end);
m3=sum(X3)'/length(find(train(:,1)==3));
X3=X3';
Sw=zeros(13); % define Sw
% calculate Sw using 4.40, 4.41 and 4.42
for i=1:length(find(train(:,1)==1))
    Sw=Sw+(X1(:,i)-m1)*(X1(:,i)-m1)';
end
for i=1:length(find(train(:,1)==2))
    Sw=Sw+(X2(:,i)-m2)*(X2(:,i)-m2)';
end
for i=1:length(find(train(:,1)==3))
    Sw=Sw+(X3(:,i)-m3)*(X3(:,i)-m3)';
end
% calculated m using 4.44
m=sum(train(:,2:end))'/length(train);
St=zeros(13); % define St
% calculate St using 4.43
for i=1:length(train)
    St=St+(train(i,2:end)'-m)*(train(i,2:end)'-m)';
end
% calculating Sb using 4.45
Sb=St-Sw;
% the concept given in section 4.1.6 is used to calculate W matrix
[A,B]=eig(inv(Sw)*Sb);
[MM,NN]=sort(diag(B),'descend');
% W matrix using eigen vector corresponding to maximum eigen values
W=[A(:,NN(1)) A(:,NN(2))]; 
Y_train=train(:,2:end)*W;
% visualize plot of Y 
% predict of the training data
output=knnclassify(Y_train,Y_train,train(:,1));
z3=find(output==train(:,1)==0); 
figure(3)
scatter(Y_train(:,1),Y_train(:,2),train(:,1)*30,T,'LineWidth',3)
hold on
colour_mat=zeros(length(z3),3); % to give proper colour
for i=1:length(z3)
    colour_mat(i,output(z3(i)))=1;
end
scatter(Y_train(z3,1),Y_train(z3,2),500*ones(length(z3),1),colour_mat,'+','LineWidth',3)
title('Training set prediction, Fisher classifier')


% calculating the prediction of Test data
Y_test=test(:,2:end)*W;
% using knn classification for determining the class
output=knnclassify(Y_test,Y_train,train(:,1));
z4=find(output==test(:,1)==0);  % points misclassified

%plot of prediction of the class
figure(4)
scatter(Y_test(:,1),Y_test(:,2),test(:,1)*30,T2,'LineWidth',3)
hold on
colour_mat=zeros(length(z4),3); % to give proper colour
for i=1:length(z4)
    colour_mat(i,output(z4(i)))=1;
end
scatter(Y_test(z4,1),Y_test(z4,2),500*ones(length(z4),1),colour_mat,'+','LineWidth',3)
title('Test set prediction, Fisher classifier')

