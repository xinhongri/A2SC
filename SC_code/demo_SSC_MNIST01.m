clear;clc;addpath(genpath('E:\sync\104-research\Code\matlab\2022AAAI\SC_code'));
Dataname='MNIST3000_2500space';load(strcat(['E:\backup\2021ICCV\MNIST\',Dataname]));
clearvars -EXCEPT Dataname trainImages trainImages_space trainLabels trainLabels_space; 
fid = fopen(strcat(['E:\Data\MNIST\SSC_',Dataname],'.txt'),'a+');
fprintf(fid,strcat('\r\n\r\n',datestr(datetime),'\r\n'));
X=trainImages';X_space=trainImages_space';
[X]=NormalizeData(double(X));[X_space]=NormalizeData(double(X_space));
cluster_num=length(unique(trainLabels));
trainLabels(trainLabels==0)=10;trainLabels_space(trainLabels_space==0)=10;
lambda=[0.000001 0.00001 0.0001 0.001 0.01 0.1 1 10];
best_acc=0;
rep = 1;

% for j = 1 : size(lambda, 2)
%     disp(j);
%     alpha = lambda(j);
%     [Z] = ssc_relaxed(X, alpha);
%     ZZ = 0.5*(abs(Z)+abs(Z'));
%     for i = 1 : rep
%         C = SpectralClustering(ZZ, cluster_num);
%         [AA nmi1(i) avgent] = compute_nmi(trainLabels_space,C);
%         acc1(i) = Accuracy(C,double(trainLabels_space));
%     end
%     fprintf(fid,['|lambda=' num2str(lambda(j))  '\r\n']);
%     fprintf(fid,['trainImages-trainLabels_space: \n']);
%     fprintf(fid,'ACC = %6.4f, std = %6.4f  NMI = %6.4f, std = %6.4f \r\n',mean(acc1),std(acc1),mean(nmi1),std(nmi1));
% end


for j = 1 : size(lambda, 2)
    disp(j);
    alpha = lambda(j);
    [Z_space] = ssc_relaxed(X_space, alpha);
    ZZ_space = 0.5*(abs(Z_space)+abs(Z_space'));
    for i = 1 : rep
        C_space = SpectralClustering(ZZ_space, cluster_num);
        [AA nmi4(i) avgent] = compute_nmi(trainLabels_space,C_space);
        acc4(i) = Accuracy(C_space,double(trainLabels_space));
    end
    fprintf(fid,['|lambda=' num2str(lambda(j))  '\r\n']);
    fprintf(fid,['trainImages_space-trainLabels_space: \n']);
    fprintf(fid,'\n ACC = %6.4f, std = %6.4f  NMI = %6.4f, std = %6.4f \r\n\r\n',mean(acc4),std(acc4),mean(nmi4),std(nmi4));
end
fclose all;
% save(strcat(['E:\Data\MNIST\SSC_',Dataname,'.mat']),'trainLabels','trainLabels_space','C','C_space');