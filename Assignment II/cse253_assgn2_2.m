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
%% Initialization
n_iter = 300;
NI = 785; NH1U = 20; NH2U = 20; NO = 10; alpha = 1/n_train; lambda = 0; gamma = 0; v1 = 0;v2 = 0; v3 = 0;
WIH1 = 2*sqrt(6/(NI + NH1U))*(rand(NI,NH1U) -0.5);
WH1H2 = 2*sqrt(6/(NH1U+NH2U+1))*(rand(NH1U+1,NH2U) - 0.5);
WH2O = 2*sqrt(6/(NH2U+1 + NO))*(rand(NH2U+1,NO) - 0.5);

train_error = zeros(n_iter,1);
vldtn_error = zeros(n_iter,1);
test_error = zeros(n_iter,1);
% Training
tic;
for iter = 1:n_iter
    A1 = (WIH1')*X; [Z1,H1_] = act(A1,'tanh'); Z1 = [ones(1,n_train);Z1];
    A2 = (WH1H2')*Z1; [Z2,H2_] = act(A2,'tanh'); Z2 = [ones(1,n_train);Z2];
    Y = exp((WH2O')*Z2);
    for sample = 1:n_train %Normalizing activations
        Y(:,sample) = Y(:,sample)/sum(Y(:,sample));
    end
    Del_O = T - Y;
    v3 = gamma*v3 - alpha*Z2*(Del_O');
    Del_H2 = H2_.*(WH2O(2:NH2U+1,:)*Del_O);
    v2 = gamma*v2 - alpha*Z1*(Del_H2');
    Del_H1 = H1_.*(WH1H2(2:NH1U+1,:)*Del_H2);
    v1 = gamma*v1 - alpha*X*(Del_H1');
    
    WIH1 = (1-alpha*lambda)*WIH1 - v1;
    WH1H2 = (1-alpha*lambda)*WH1H2 - v2;
    WH2O = (1-alpha*lambda)*WH2O - v3;
    %Calculating training error
    [~,I] = max(Y); train_output = (I-1)';
    train_error(iter) = sum(1-(train_output == train_labels))/n_train;
    %Calculating validation error
    A1 = (WIH1')*vldtn_images; Z1 = act(A1,'tanh'); Z1 = [ones(1,n_vldtn);Z1];
    A2 = (WH1H2')*Z1; Z2 = act(A2,'tanh'); Z2 = [ones(1,n_vldtn);Z2];
    [~,I] = max((WH2O')*Z2); vldtn_output =(I-1)';
    vldtn_error(iter) = sum(1-(vldtn_output == vldtn_labels))/n_vldtn;
    %Calculating test error
    A1 = (WIH1')*test_images; Z1 = act(A1,'tanh'); Z1 = [ones(1,n_test);Z1];
    A2 = (WH1H2')*Z1; Z2 = act(A2,'tanh'); Z2 = [ones(1,n_test);Z2];
    [~,I] = max((WH2O')*Z2); test_output = (I-1)';
    test_error(iter) = sum(1-(test_output == test_labels))/n_test;
end
toc;
hold on
plot(vldtn_error,'g');xlabel('Iteration No');ylabel('Error Rate');
plot(test_error,'r');
plot(train_error,'b');legend('Validation Set','Test Set','Training Set');
[mvl,loc] = min(vldtn_error);
plot(loc,mvl,'g*');
title(['\lambda =',num2str(lambda),' \gamma =',num2str(gamma),' \alpha =',num2str(alpha)]);
hold off;