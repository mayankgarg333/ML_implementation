

clc
close all
clear all

%% training data and model training

F=[];
for k=[25:5:90]
    load(['profile_1_features_' num2str(k)])

    for i=1:length(features)
        F=[F;features{i}];
    end
end

X=F(:,1:7);
Y=F(:,end);


%% test data and testing


F=[];
for k=[60:5:90]
    load(['data_profile_1_12ah_features_' num2str(k)])

    for i=1:length(features)
        F=[F;features{i}];
    end
end
    
Xnew=F(:,1:7);
Ynew=F(:,end);
label = knnclassify(Xnew, X, Y,3);

%%
fig=figure();
set(fig,'color',[1 1 1])
subplot(1,2,1)
simple = label;
plot(simple,'Linewidth',2)
hold on
plot(Ynew,'r--','Linewidth',2)
m=mean(abs(label-Ynew))*100;
m = round((m*100))/100;
text(250,0.55,{'Mean SOC' ['Error ' num2str(m) '%']})

ylim([0.5 1])
set(findall(fig,'-property','FontSize'),'FontSize',16)
legend('Predicted SOC','Actual SOC')
title({'12Ah Battery SOC prediction', 'using 10Ah battery data'})
xlabel('Data Points')
ylabel('SOC')
set(gca,'LineWidth',2)



% max(abs(simple-Ynew))*100

%% test data and testing


F=[];
for k=[60:5:90]
    load(['data_profile_1_8ah_features_' num2str(k)])

    for i=1:length(features)
        F=[F;features{i}];
    end
end
    
Xnew=F(:,1:7);
Ynew=F(:,end);
label = knnclassify(Xnew, X, Y,3);

%%
subplot(1,2,2)
simple = label;
plot(simple,'Linewidth',2)
hold on
plot(Ynew,'r--','Linewidth',2)
m=mean(abs(label-Ynew))*100;
m = round((m*100))/100;
text(250,0.55,{'Mean SOC' ['Error ' num2str(m) '%']})
ylim([0.5 1])
set(findall(fig,'-property','FontSize'),'FontSize',16)
legend('Predicted SOC','Actual SOC')
title({'8Ah Battery SOC prediction', 'using 10Ah battery data'})
xlabel('Data Points')
ylabel('SOC')
set(gca,'LineWidth',2)

set(fig, 'Position', [300 0 1500 600])
