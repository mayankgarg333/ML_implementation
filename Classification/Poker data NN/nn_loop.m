result=[];
save result.mat result
for k=35:60
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load poker_train.data
load poker_test.data
load 'result.mat'
poker=[poker_train;poker_test];
poker(:,end)=poker(:,end)+1;

train1 = []; %stores the training samples
test = [];  %stores the test samples
for i=1:10 % changed to (8-classes:7) for reverse classes 
    ind{i} = find(poker(:,end)==i);
    len = length(ind{i});
    t = randperm(len);
    half = round(len/10);
    train1 = [train1; poker(ind{i}(t(1:half)), :)];
    test = [test; poker(ind{i}(t(half+1:end)), :)];
end

train_mat=train1(:,1:end-1);
train_y=train1(:,end);
classes=10;
B=train_y*ones(1,classes);
D=ones(length(train_y),1)*[1:classes];
train_y=(B==D);

test_mat=test(:,1:end-1);
test_y=test(:,end);
classes=10;
B=test_y*ones(1,classes);
D=ones(length(test_y),1)*[1:classes];
test_y=(B==D);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


x = train_mat';
t = train_y';
% Create a Pattern Recognition Network
hiddenLayerSize = k
net = patternnet(hiddenLayerSize);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;


% Train the Network
%net.trainFcn = 'trainbr';   % got out of the memory.. try more memory
%system
%net.trainFcn = 'trainscg';   % exactly same as trainlm (default)
[net,tr] = train(net,x,t);         % look for different train functions 

% Ttain set error
y = net(x);
e = gsubtract(t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);
performance = perform(net,t,y);

x_t=test_mat';
t_t=test_y';
%test set error
y_t = net(x_t);
e_t = gsubtract(t_t,y_t);
tind_t = vec2ind(t_t);
yind_t = vec2ind(y_t);
percentErrors_t = sum(tind_t ~= yind_t)/numel(tind_t);
performance_t = perform(net,t_t,y_t);
result=[result ; hiddenLayerSize (1-percentErrors)*100 (1-percentErrors_t)*100]
save result.mat result
if (1-percentErrors_t)*100 >100
    break
end
clear all
end

a=5
load poker_train.data
load poker_test.data
poker=[poker_train;poker_test];
poker(:,end)=poker(:,end)+1;
test = poker;  %stores the test samples
test_mat=test(:,1:end-1);
test_y=test(:,end);
classes=10;
B=test_y*ones(1,classes);
D=ones(length(test_y),1)*[1:classes];
test_y=(B==D);

x_t=test_mat';
t_t=test_y';
y_t = net(x_t);
e_t = gsubtract(t_t,y_t);
tind_t = vec2ind(t_t);
yind_t = vec2ind(y_t);
percentErrors_t = sum(tind_t ~= yind_t)/numel(tind_t);
performance_t = perform(net,t_t,y_t);
final_result=[hiddenLayerSize (1-percentErrors)*100 (1-percentErrors_t)*100]

