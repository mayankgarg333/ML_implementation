clc
close all
clear all

%% Load the data
F=[];
for k=[1:1:500]
    load(['healthy battery data features\profile_features_' num2str(k)])
    Tmp=[];
    for i=1:length(features)
        Tmp=[Tmp;features{i}];
    end
    A=mean(Tmp);
    F=[F;A(1:end-2)];
end

for k=[1:1:100]
    load(['Unhealthy battery data features 1\profile_features_' num2str(k)])
    Tmp=[];
    for i=1:length(features)
        Tmp=[Tmp;features{i}];
    end
    A=mean(Tmp);
    F=[F;A(1:end-2)];
end

for k=[1:1:100]
    load(['Unhealthy battery data features 2\profile_features_' num2str(k)])
    Tmp=[];
    for i=1:length(features)
        Tmp=[Tmp;features{i}];
    end
    A=mean(Tmp);
    F=[F;A(1:end-2)];
end

%% PCA of input matrix

F=(F - ones(length(F),1)*min(F)) ./ (ones(length(F),1)*( max(F) - min(F) ));
P=pca(F);
Out=F*P(:,[1:2]);
fig=figure(1); 
set(fig, 'Color', [1 1 1])
scatter(Out(1:500,1),Out(1:500,2),[],'b')
hold on
scatter(Out(501:600,1),Out(501:600,2),[],'r')
hold on
scatter(Out(601:700,1),Out(601:700,2),[],'g')
title('Data Visualization') 
legend('Healthy','Unhealthy 1' ,'Unhealthy 2')
box on
set(findall(gcf,'-property','FontSize'),'FontSize',16)
set(fig, 'Position', [300 300 700 500])
set(gca,'LineWidth',2)


%% Kmeans on the data after PCA

[idx,C] = kmeans(Out,3);

fig=figure(2); 
set(fig, 'Color', [1 1 1])
plot(Out(idx==1,1),Out(idx==1,2),'r.','MarkerSize',12)
hold on
plot(Out(idx==2,1),Out(idx==2,2),'b.','MarkerSize',12)
plot(Out(idx==3,1),Out(idx==3,2),'g.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','NW')
title('Data Clustering using K-means') 
box on
set(findall(gcf,'-property','FontSize'),'FontSize',16)
set(fig, 'Position', [300 300 700 500])
set(gca,'LineWidth',2)




%% Linkage clustering after PCA

Z = linkage(Out);
c = cluster(Z,'maxclust',3);
fig=figure(3);
set(fig, 'Color', [1 1 1])
h=scatter(Out(:,1),Out(:,2),[],c);
title('Data Clustering using Linkage Clustering') 
box on
set(findall(gcf,'-property','FontSize'),'FontSize',16)
set(fig, 'Position', [300 300 700 500])
set(gca,'LineWidth',2)

fig=figure(4);
set(fig, 'Color', [1 1 1])
H = dendrogram(Z,0,'ColorThreshold','default');
set(H,'LineWidth',2)
% set(gca,'XTick',[])
title('Dendrogram') 
box on
set(findall(gcf,'-property','FontSize'),'FontSize',16)
set(fig, 'Position', [300 300 700 500])
set(gca,'LineWidth',2)
