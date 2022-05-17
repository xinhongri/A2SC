%%
% 选取前num张图片，共10类，每类num/10张
clear;clc;
% load('E:\Data\MNIST\MNIST.mat');
num=2000;J=2000;
% clear trainImages0 trainImages1 trainLabels0 trainLabels1 testImages testImages0 testImages1 testLabels testLabels0 testLabels1;
% trainImages_select=zeros(28,28,num);
% trainLabels_select=zeros(num,1);
% flag=1;j=1;
% folder = strcat('E:\Data\MNIST\train',num2str(num),'\');
% if ~exist(folder,'dir')
%     mkdir(folder);
% end
%  while flag<=10
%      count = 1;
%      while count<=(num/10)
%         for i = 1:60000
%             if trainLabels(i,1)==mod(flag,10)
%                 img=reshape(trainImages(:, :, i), 28, 28);
%                 %可以转换为任意图像格式.jpg .png .bmp
% %                 imwrite(img, strcat(folder ,num2str(flag),'_',num2str(i),'.png'));
%                 trainImages_select(:, :, j)=trainImages(:, :, i);
%                 trainLabels_select(j)=trainLabels(i);
%                 j=j+1;
%                 count = count+1;
%             end
%             if count==(num/10+1)
%                 disp(['flag:',int2str(flag),'count:',int2str(count)]);
%                 break;
%             end
%         end
%     end
%     flag = flag+1;
%  end
%  imshow(trainImages_select(:,:,num));
%  trainImages=uint8((reshape(trainImages_select,[784,num]))');
%  trainLabels=uint8(trainLabels_select);
%  clear count flag i img j trainImages_select trainLabels_select trainLabels_select1 folder;
%  save(strcat('E:\Data\MNIST\MNIST',num2str(num),'.mat'));
 
% clearvars -EXCEPT num J;clc;load(strcat('E:\Data\MNIST\MNIST',num2str(num),'.mat'));
% trainLabels(find(trainLabels==0))=10;
% trainLabels_swap=uint8(repmat([1:10]',num/10,1)); %order change
% trainImages_10000784=trainImages;
% trainImages_10000785=[trainImages_10000784,trainLabels];
% trainImages_sort=sortrows(trainImages_10000785,785);
% trainLabels_sort=trainImages_sort(:,785:785);
% trainImages_sort=trainImages_sort(:,1:784);   %10000*784
% flag=1;i=1;
% folder = strcat('E:\Data\MNIST\train',num2str(num),'\');
% if ~exist(folder,'dir')
%     mkdir(folder);
% end
% while i<=num
%     img=reshape(trainImages_sort(i, :), 28, 28);
%     %可以转换为任意图像格式.jpg .png .bmp
%     imwrite(img, strcat(folder ,num2str(i),'_',num2str(trainLabels(i)),'.png'));
%     if mod(i,100)==0
%         disp(i);
%     end
%     i=i+1;
% end
% 
% sta_sort=tabulate(trainLabels_sort);
% %% 选J个图片，每类是J/10张图片
% trainImages_select=single(zeros(J,784));
% trainLabels_select=single(zeros(J,1));
% folder=strcat('E:\Data\MNIST\train_J',num2str(J),'\');
% if ~exist(folder,'dir')
%     mkdir(folder);
% end
% for i =1:10
%     Index=find(trainLabels_sort==i);
%     trainImages_select(i*J/10-J/10+1:i*J/10,:)=trainImages_sort(Index(1:J/10),:);
%     trainLabels_select(i*J/10-J/10+1:i*J/10,1)=trainLabels_sort(Index(1:J/10),1);
% end
% trainImages_J=uint8(trainImages_select);
% trainLabels_J=uint8(trainLabels_select);
% sta_J=tabulate(trainLabels_J);
% for i=1:J
%     imwrite(reshape(trainImages_J(i,:),[28,28]),strcat(folder,num2str(i),'.png'));
%     if mod(i,100)==0
%         disp(i);
%     end
% end
% clear trainImages_select trainLabels_select;
% clear Index flag i img j folder trainImages_10000784 trainImages_10000785;
% save(strcat('E:\Data\MNIST\MNISTtrain',num2str(num),'_',num2str(J),'J.mat'));


%% 投影矩阵
load(strcat('E:\Data\MNIST\MNISTtrain',num2str(num),'_',num2str(J),'J.mat'));
% 假设每类都聚类正确，求space
ii=5;diff_alpha=1;
trainImages_Jsingle=single(trainImages_J');  % 节省空间不用double，400*1024—1024*400
train_base(784,10)=single(0);
test_vector(784,J/10-1,10)=single(0);
test_tran(784,784,10)=single(0);
trainImages_allspace(num,10,784)=single(0);
trainImages_new(784,num)=single(0);

tic;
for i = 1:10  % 类别数
    train_base(:,i)=trainImages_Jsingle(:,(J/10)*i-(J/10)+1);
    flag=1;
    for j = (J/10)*i-(J/10)+2:(J/10)*i
        test_vector(:,flag,i) =trainImages_Jsingle(:,j)-train_base(:,i);
        flag=flag+1;
    end
    test_tran(:,:,i)=test_vector(:,:,i)*inv(test_vector(:,:,i)'*test_vector(:,:,i)) * test_vector(:,:,i)';
end
toc;
disp('投影矩阵计算完成！');



trainImages_sort=single(reshape(trainImages_sort',784,num));
tic;
for i = 1:num
    for j = 1:10
        trainImages_allspace(i,j,:)=(test_tran(:,:,j)*(trainImages_sort(:,i)-train_base(:,j)) + train_base(:,j))';
    end
%     if mod(i,100)==0
%         disp(i);
%     end
end
toc;
trainImages_allspace=permute(trainImages_allspace,[3,2,1]); % 数据、第几类、第几张图片400,40,1024—1024,40,400
disp('投影点计算完成！');

tic;
%% 攻击
for i = 1:num
    j = trainLabels_swap(i);
    if j==0
        trainImages_new(:,i) = trainImages_allspace(:,10,i);
    else
        trainImages_new(:,i) = trainImages_allspace(:,j,i);
    end
%     if mod(i,100)==0
%         disp(i);
%     end
end
toc;

train_diff=trainImages_new-trainImages_sort;
adv=trainImages_sort+diff_alpha*train_diff;
trainImages_space=uint8(adv');
trainImages_new=uint8(trainImages_new);
subplot(411),imshow(reshape(trainImages_sort(:,ii),28,28));
subplot(412),imshow(reshape(trainImages_space(ii,:),28,28));
subplot(413),imshow(reshape(trainImages_new(:,ii), 28, 28));
subplot(414),imshow(reshape(trainImages_sort(:,ii*(num/10)-(num/10)+1),28,28));
if  diff_alpha==1
    folder= strcat('E:\Data\MNIST\trainspace',num2str(num),'\');
else
    folder= strcat('E:\Data\MNIST\trainspace',num2str(num),'_0',num2str(diff_alpha*100),'\');
end
if ~exist(folder,'dir')
    mkdir(folder);
end
% for i =1:num
%     imwrite(reshape(trainImages_space(i,:),[28,28]), strcat(folder,num2str(i),'_',num2str(trainLabels_sort(i)),'_',num2str(trainLabels_swap(i)),'.png'));
% end
diff_space=single(trainImages_sort')-single(reshape(trainImages_space,num,784));
infinity=max(abs(diff_space),[],2)/255;
mean_space=sum(infinity)/num;

mean_infinity=mean_space;
Euclidean=sqrt(sum(diff_space.*diff_space,2))/255;
mean_Euclidean=sum(Euclidean)/num;
trainLabels_space=trainLabels_swap;
trainImages_allspace=uint8(trainImages_allspace);
clear i j flag ii folder trainImages_new trainImages_single;
clear adv train_base test_diff test_tran test_vector diff_space;
%节省空间，可删除这一行
if  diff_alpha==1
    save(strcat('E:\Data\MNIST\MNIST',num2str(num),'_',num2str(J),'space.mat'));
else
    save(strcat('E:\Data\MNIST\MNIST',num2str(num),'_',num2str(J),'_0',num2str(diff_alpha*100),'space.mat'));
end
% 时间已过 0.056666 秒。
% 投影矩阵计算完成！
% 时间已过 29.001383 秒。
% 投影点计算完成！
% 时间已过 0.007777 秒。