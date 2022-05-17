%% 人工数据生成
clear;close;clc;
mu1 = [80 80];SIGMA1 = [500 0; 0 500];
r1 = mvnrnd(mu1,SIGMA1,100);
plot(r1(:,1),r1(:,2),'r+');

hold on;
mu2 = [175 175];SIGMA2 = [1000 0; 0 1000];
r2 = mvnrnd(mu2,SIGMA2,100);
plot(r2(:,1),r2(:,2),'*')

L=ones(100,1);
trainImages=[r1', r2'];
trainLabels=[L; L*2];

save('E:\Data\Artificial\Artificial200_1.mat','trainImages','trainLabels');