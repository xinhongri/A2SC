%% 人工数据生成
clear;close;clc;
r1 = [0,0,0; 0,5,0; 5,0,0; 5,5,5];
plot3(r1(:,1),r1(:,2),r1(:,3),'rd');

hold on;
r2 =  [30,20,20; 20,20,30; 20,30,20; 30,30,30];
plot3(r2(:,1),r2(:,2),r2(:,3),'*')


trainImages=[r1', r2'];
trainLabels=[1;1;1;1;2;2;2;2];

save('E:\Data\Artificial\Artificial4_1.mat','trainImages','trainLabels');