%% Loading Images
trainl_images = zscore(loadMNISTImages('train-images.idx3-ubyte'));
trainl_labels = loadMNISTLabels('train-labels.idx1-ubyte');

n_train = 50000; n_vldtn = 10000;n_test = 10000;
X = [ones(1,n_train);trainl_images(:,1:n_train)]; %X = train_images
train_labels = trainl_labels(1:n_train);
T = zeros(10,n_train); %Target matrix.
for sample = 1:n_train
    T(train_labels(sample)+1,sample) = 1;% Each column has a '1' in the location of the true class of that image
end

vldtn_images = [ones(1,n_vldtn);trainl_images(:,50001:60000)];
vldtn_labels = trainl_labels(50001:60000);
clear trainl_images trainl_labels

test_images = zscore(loadMNISTImages('t10k-images.idx3-ubyte'));
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
test_images = [ones(1,n_test);test_images];
%%
% Initialization
n_iter = 300;
NI = 785; NHU = 80; NO = 10; alpha = 1/n_train; lambda = 0; gamma = 0; v1 = 0;v2 = 0;
WIH = 2*sqrt(6/(NI + NHU))*(rand(NI,NHU) - 0.5); WHO = 2*sqrt(6/(NHU+1 + NO))*(rand(NHU+1,NO) - 0.5);
actfun = 'tanh';
train_error = zeros(n_iter,1);
vldtn_error = zeros(n_iter,1);
test_error = zeros(n_iter,1);
% Training
tic;
for iter = 1:n_iter
    A = (WIH')*X;
    [Z,H_] = act(A,actfun);
    Z = [ones(1,n_train);Z]; Y = exp((WHO')*Z);
    for sample = 1:n_train %Normalizing activations
        Y(:,sample) = Y(:,sample)/sum(Y(:,sample));
    end
    Del_k = T - Y;
    v1 = gamma*v1 - alpha*X*((H_.*(WHO(2:NHU+1,:)*Del_k))');
    v2 = gamma*v2 - alpha*Z*(Del_k');
    WIH = (1-alpha*lambda)*WIH - v1;
    WHO = (1-alpha*lambda)*WHO - v2;
    %Calculating training error
    [~,I] = max(Y); train_output = (I-1)';
    train_error(iter) = sum(1-(train_output == train_labels))/n_train;
    %Calculating validation error
    [~,I] = max((WHO')*[ones(1,n_vldtn);act((WIH')*vldtn_images,actfun)]);
    vldtn_output =(I-1)'; vldtn_error(iter) = sum(1-(vldtn_output == vldtn_labels))/n_vldtn;
    %Calculating test error
    [~,I] = max((WHO')*[ones(1,n_test);act((WIH')*test_images,actfun)]);
    test_output = (I-1)'; test_error(iter) = sum(1-(test_output == test_labels))/n_test;
end
toc;
hold on;
plot(vldtn_error,'g');xlabel('Iteration No');ylabel('Error Rate');
plot(test_error,'r');
plot(train_error,'b');legend('Validation Set','Test Set','Training Set');
[mvl,loc] = min(vldtn_error);
plot(loc,mvl,'g*');
title(['\lambda =',num2str(lambda),' \gamma =',num2str(gamma),' \alpha =',num2str(alpha)]);
hold off;
[mvl,loc,vldtn_error(300),test_error(300),test_error(loc),max(abs(WIH(:))),max(abs(WHO(:)))]