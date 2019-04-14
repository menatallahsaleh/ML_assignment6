close all
clear all
clc
[num,~]=xlsread('house_prices_data_training_data.csv');
y=num(:,3);
%y=(y-mean(y))/std(y);
features=num(:,4:end);
m=length(features(1,:));
n=18;
%To normalize features
x=features;
%x=zeros(length(features(:,1)),length(features(1,:)));
% for f=1:length(features(1,:))
%     x(:,f)=(features(:,f)-mean(features(:,f)))/std(features(:,f));
% end
%To calculate covariance matrix
Corr_x = corr(x);
cov_x=cov(x);
[U,S,V]=svd(cov_x);
k=0;
alpha=100;
while (alpha>0.001) && k<n
     k=k+1;
    alpha=1-sum(max(S(:,1:k)))/sum(max(S));
end
U_reduced=U(:,1:k);
z=U_reduced'*x';
x_approx=z'*U_reduced';
error=((1/m)*sum((x_approx-x)'))';
% linear regression
[ hyp_1,hyp_2,hyp_3,hyp_4 ] = Hypotheses( z' );
hyp=hyp_1;
thetas_old=rand(length(hyp(1,:)),1);%ones(1,length(variables(1,:))+1);
thetas_new=zeros(length(hyp(1,:)),1);
Alpha=100;
iter=10000;
mse=zeros(1,iter);
 for k=1:iter
        % to calculate hypothesis values 
       h=hyp*thetas_old;
       diff=h-y;
       thetas_new= thetas_old-(Alpha/length(x_approx(:,1)))*transpose(hyp)*diff; 
        thetas_old=thetas_new;
 end
 y_estimate=hyp*thetas_new;
 
 %% K-clustring
 K_tot=[7];
 e_tot=[];
 for k=1:length(K_tot)
     K=K_tot(k);
 m_2=length(features(:,1));
 pos=round((m_2-1)*rand(K,1)+1);
 miu_new=features(pos,:);
 miu_old=zeros(length(miu_new(:,1)),length(miu_new(1,:)));
 data_classes=zeros(m_2,1);
 while sum(sum(abs(miu_old-miu_new)))>0
      miu_old=miu_new;
 for i=1:m_2
     d_tot=[];
     for j=1:K
         d=sqrt(sum(((features(i,:)-miu_new(j,:)).^2)));
         d_tot=[d_tot d];
     end
     non_zero_elements=find(d_tot>0);
     [~,p]=min(d_tot(non_zero_elements));
     data_classes(i)=p;
 end
 % To update mius
 for c=1:K
 miu_new(c,:)=mean(features(find(data_classes==c),:));
 end
 end
 %to calculate error
 e=0;
 for i=1:m_2
 e=e+sum((features(i,:)-miu_old(data_classes(i),:)).^2);
 end
 e_tot=[e_tot e/m_2];
 end
 %% 
  K=7;
 m_2=length(x_approx(:,1));
 pos=round((m_2-1)*rand(K,1)+1);
 miu_new=x_approx(pos,:);
 miu_old=zeros(length(miu_new(:,1)),length(miu_new(1,:)));
 data_classes_2=zeros(m_2,1);
 while sum(sum(abs(miu_old-miu_new)))>0
      miu_old=miu_new;
 for i=1:m_2
     d_tot=[];
     for j=1:K
         d=sqrt(sum(((x_approx(i,:)-miu_new(j,:)).^2)));
         d_tot=[d_tot d];
     end
     non_zero_elements=find(d_tot>0);
     [~,p]=min(d_tot(non_zero_elements));
     data_classes_2(i)=p;
 end
 % To update mius
 for c=1:K
 miu_new(c,:)=mean(x_approx(find(data_classes_2==c),:));
 end
 end
 %to calculate error
 e=0;
 for i=1:m_2
 e=e+sum((x_approx(i,:)-miu_old(data_classes_2(i),:)).^2);
 end
 e=e/m_2;
 %% Anomoly_Detection
 epsilon=10^-10;
 mean_features=mean(features);
 std_features=std(features);
 f=[];
 joint_pdf=1;
 for i=1:m
 f1=normcdf(features(:,i),mean_features(1,i),std_features(1,i));
 joint_pdf=joint_pdf.*f1;
 f=[f f1];
 end
 anomoly_data=find(joint_pdf<epsilon);
 anomoly_data=[anomoly_data;find(joint_pdf>(1-epsilon))];