%% start code for project #1: linear regression
%pattern recognition, CSE583/EE552
%Weina Ge, Aug 2008

% imp note - only y_predit is extra plotted, rest of the plots have
% the same meaning as given, only the blue plot was tuned in green 
% for better viewing

clear 
close all

%load the data points
load data10.mat

%plot the groud truth curve
figure(1)
xx = linspace(1,4*pi,50);
yy = sin(.5*xx);
plot(xx,yy,'g-');
hold on
plot(x,y,'o','MarkerSize',3);
%plot the noisy observations
plot(x,t,'ro','MarkerSize',3);
xlabel('x')
ylabel('t')

%% PART 1 curve fit for M-5

M=5;
X=[];
X_plot=[];
for i=0:M
    X=[X x'.^i];
    X_plot=[X_plot xx'.^i];
end
w=inv(X'*X)*X'*t';
y_predict = X_plot*w ;
plot(xx,y_predict,'m-');
title('M=5,curve fitting')



%% PART 1 curve fit for different Ms
figure(2)
W=[];
for M=0:9
X=[];
X_plot=[];
for i=0:M
    X=[X x'.^i];
    X_plot=[X_plot xx'.^i];
end
% calculation of the w vector as given in notes
w=inv(X'*X)*X'*t';
% storing w for the report
W=[W,[w;NaN(9-M,1)]];
y_predict = X_plot*w ;  
%plots 
subplot(2,5,M+1)
plot(xx,y_predict,'m-','LineWidth',3);
hold on
plot(xx,yy,'g-');
plot(x,y,'o','MarkerSize',3);
plot(x,t,'ro','MarkerSize',3);
xlabel('x')
ylabel('t')
title(['N=10, M=',num2str(M)])
end
W2=W;

%% PART 1 curve fit for different data set
figure(3)
M=9;
W=[];
for k =1:6
npts = 10*k; 
name=['data',num2str(npts),'.mat'];% loading diffenernt data file
load(name)
X=[];
X_plot=[];
for i=0:M
    X=[X x'.^i];
    X_plot=[X_plot xx'.^i];
end
% calculation of w as given in notes
w=inv(X'*X)*X'*t';
W=[W,[w;NaN(9-M,1)]];  % storing w
y_predict = X_plot*w ; % rediction using w
%plots
subplot(2,3,k)
plot(xx,y_predict,'m-','LineWidth',3);
hold on
plot(xx,yy,'g-');
plot(x,y,'o','MarkerSize',3);
plot(x,t,'ro','MarkerSize',3);
xlabel('x')
ylabel('t')
title(['N=',num2str(npts),', M=9'])


end
W3=W;

%% PART 2 add regularization 

lambda=exp(-1);
load data10.mat
figure(4)
W=[];
for M=0:9
X=[];
X_plot=[];
for i=0:M
    X=[X x'.^i];
    X_plot=[X_plot xx'.^i];
end
% calculation of w as given in the notes
w=inv(X'*X+lambda*eye(M+1))*X'*t';
W=[W,[w;NaN(9-M,1)]];% storing w
y_predict = X_plot*w ; % y prediction
%plots
subplot(2,5,M+1)
plot(xx,y_predict,'m-','LineWidth',3);
hold on
plot(xx,yy,'g-');
plot(x,y,'o','MarkerSize',3);
plot(x,t,'ro','MarkerSize',3);
xlabel('x')
ylabel('t')
title(['ln lambda = -1, M=',num2str(M)])
end
W4=W;


%% PART 2 same M, same data, Different lambda

M=5;
load data10.mat
figure(5)
W=[];
L=[exp(-18),exp(-10),exp(-5),exp(0),exp(5),exp(10)]; % differnt lambda values
for k=1:6
lambda=L(k);
X=[];
X_plot=[];
for i=0:M
    X=[X x'.^i];
    X_plot=[X_plot xx'.^i];
end
% calculation of w as given in notes
w=inv(X'*X+lambda*eye(M+1))*X'*t';
W=[W,[w;NaN(9-M,1)]]; % storing w
y_predict = X_plot*w ;  % predicting y
%plots
subplot(2,3,k)
plot(xx,y_predict,'m-','LineWidth',3);
hold on
plot(xx,yy,'g-');
plot(x,y,'o','MarkerSize',3);
%plot the noisy observations
plot(x,t,'ro','MarkerSize',3);
xlabel('x')
ylabel('t')
title(['M=5, lambda=',num2str(L(k))])
end
W5=W;

%% PART 2, M =9 lambda=exp(-10), Different Data Points 

M=9;
W=[];
figure(6)
lambda=exp(-10);
for k =1:6
npts = 10*k; 
name=['data',num2str(npts),'.mat'];
load(name)
X=[];
X_plot=[];
for i=0:M
    X=[X x'.^i];
    X_plot=[X_plot xx'.^i];
end
% calculation of w as given in notes
w=inv(X'*X+lambda*eye(M+1))*X'*t';
W=[W,[w;NaN(9-M,1)]]; % string w 
y_predict = X_plot*w ;  % y prediction
%plots
subplot(2,3,k)
plot(xx,y_predict,'m-','LineWidth',3);
hold on
plot(xx,yy,'g-');
plot(x,y,'o','MarkerSize',3);
plot(x,t,'ro','MarkerSize',3);
xlabel('x')
ylabel('t')
title(['M=9, ln lambda=-10, N=',num2str(npts)])
end
W6=W;

%% PART 3 different M 

load data10.mat
figure(7)
W=[];
for M=0:9
X=[];
X_plot=[];
for i=0:M
    X=[X x'.^i];
    X_plot=[X_plot xx'.^i];
end
% clculation of we as per notes
w=inv(X'*X)*X'*t';
W=[W,[w;NaN(9-M,1)]]; % storing w
y_predict = X_plot*w; % prediction of y
beta=10/((t'-X*w)'*(t'-X*w));  % calculation of beta as given
% using the beta, variance is calcuted and 95 percent 
% confidence interval is plot
% higher and lower limit of y_prict is calcluted using the 
% 95 percent confdence interval method
y_predict_u= X_plot*w+1.96*sqrt(1/(beta*10)); % upper bound of 95 CI
y_predict_l= X_plot*w-1.96*sqrt(1/(beta*10)); % lower bound of 95 CI
%plots
subplot(2,5,M+1)
ciplot(y_predict_l,y_predict_u,xx,[0 1 0])
hold on
plot(xx,y_predict,'m-','LineWidth',3);
plot(xx,yy,'g-');
plot(x,y,'o','MarkerSize',3);
plot(x,t,'ro','MarkerSize',3);
xlabel('x')
ylabel('t')
title(['N=10, M=',num2str(M)])
end
W7=W;

%% PART 3 Different data points

figure(8)
W=[];
M=9;
for k =1:6
npts = 10*k; 
name=['data',num2str(npts),'.mat']; % loaading different data files
load(name)
X=[];
X_plot=[];
for i=0:M
    X=[X x'.^i];
    X_plot=[X_plot xx'.^i];
end
% calculation of w as given in the notes
w=inv(X'*X)*X'*t';
W=[W,[w;NaN(9-M,1)]]; % storing w
y_predict = X_plot*w ;  % y prediction 
beta=10/((t'-X*w)'*(t'-X*w)); % beta calculation as given
y_predict_u= X_plot*w+1.96*sqrt(1/(beta*npts)); % upper bound of 95 CI
y_predict_l= X_plot*w-1.96*sqrt(1/(beta*npts)); % lower bound of 95 CI
%plots
subplot(2,3,k)
ciplot(y_predict_l,y_predict_u,xx,[0 1 0])
hold on
plot(xx,y_predict,'m-','LineWidth',3);
plot(xx,yy,'g-');
plot(x,y,'o','MarkerSize',3);
plot(x,t,'ro','MarkerSize',3);
xlabel('x')
ylabel('t')
title(['M=9, N=',num2str(npts)])
end
W8=W;

%% PART 4 different M
load data10.mat
figure(9)
W=[];
beta_v=linspace(.9,8,10); % assumed beta values
alpha=.005;
for M=0:9
X=[];
X_plot=[];
beta=beta_v(M+1);
for i=0:M
    X=[X x'.^i];
    X_plot=[X_plot xx'.^i];
end
% w calcuation according the method given in notes
w=beta*inv(beta*X'*X+alpha*eye(M+1))*X'*t';
W=[W,[w;NaN(9-M,1)]]; % storing the w values
y_predict = X_plot*w; % y prediction
% calculation of S inverse
S_inv=alpha*eye(M+1)+beta*X_plot'*X_plot;
S=inv(S_inv); % calculation S
% calculation of sigma square
sigma_sq=(1/beta)+diag(X_plot*S*X_plot'); 
% using the sigma square 95 percent confidence interval is calculated
y_predict_u= X_plot*w+1.96*sqrt(sigma_sq/10); % Upper limit of 95 CI
y_predict_l= X_plot*w-1.96*sqrt(sigma_sq/10); % Lower limit of 95 CI
% plots
subplot(2,5,M+1)
ciplot(y_predict_l,y_predict_u,xx,[0 1 0])
hold on
plot(xx,y_predict,'m-','LineWidth',3);
plot(xx,yy,'g-');
plot(x,y,'o','MarkerSize',3);
plot(x,t,'ro','MarkerSize',3);
xlabel('x')
ylabel('t')
title(['alpha=.005, beta=',num2str(beta),' M=',num2str(M)])
end
W9=W;

%% PART 4 Different data
figure(10)
M=9;
W=[];
beta_v=linspace(.9,8,10); % assumed beta values
alpha=.005;
for k =1:6
npts = 10*k; 
name=['data',num2str(npts),'.mat'];
load(name)
X=[];
X_plot=[];
for i=0:M
    X=[X x'.^i];
    X_plot=[X_plot xx'.^i];
end
beta=beta_v(M+1);
% w calcuation according the method given in notes
w=beta*inv(beta*X'*X+alpha*eye(M+1))*X'*t';
W=[W,[w;NaN(9-M,1)]]; % storing the w values
y_predict = X_plot*w ; % y prediction
beta=10/((t'-X*w)'*(t'-X*w));
% calculation of S inverse
S_inv=alpha*eye(M+1)+beta*X_plot'*X_plot;
S=inv(S_inv); % calculation S
% calculation of sigma square
sigma_sq=(1/beta)+diag(X_plot*S*X_plot');
% using the sigma square 95 percent confidence interval is calculated
y_predict_u= X_plot*w+1.96*sqrt(sigma_sq/10); % Upper limit of 95 CI
y_predict_l= X_plot*w-1.96*sqrt(sigma_sq/10); % Lower limit of 95 CI
% plots 
subplot(2,3,k)
ciplot(y_predict_l,y_predict_u,xx,[0 1 0])
hold on
plot(xx,y_predict,'m-','LineWidth',3);
plot(xx,yy,'g-');
plot(x,y,'o','MarkerSize',3);
plot(x,t,'ro','MarkerSize',3);
xlabel('x')
ylabel('t')
title(['alpha=.005, M= 9, beta=',num2str(beta),' N=',num2str(npts)])
end
W10=W;

%%

