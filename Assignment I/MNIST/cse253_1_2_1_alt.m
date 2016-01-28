W = zeros(10,785);
classifier_err = zeros(10,1);
for class_lbl = 0:9
    class_lbl
    converged = 0; prev_err = 1; iter_no = 0;
    eta = init_eta;
    while eta > 1/(16*n_train)
        iter_no = iter_no + 1;
        %training
        y = (1./(1 + exp(-W(class_lbl+1,:)*train_images)))';
        t = double((train_labels == class_lbl));
        err_sum = train_images*(y-t);
        W(class_lbl+1,:) = W(class_lbl+1,:) - eta*err_sum';
        if iter_no > min_iter
            %validation
            dec_vec = W(class_lbl+1,:)*test_images;
            dec = (1./(1 + exp(dec_vec))) > 0.5;
            trgt = (test_labels == class_lbl);
            classifier_err(class_lbl+1) = sum(1 - abs(dec'-trgt))/n_test;
            if prev_err - classifier_err(class_lbl+1) < 0.01
                eta = eta/2;
            end
            prev_err = classifier_err(class_lbl+1);
        end
    end
end

dec_mat = W*test_images;
[~,I] = max(dec_mat,[],1);
test_output = (I - 1)';
test_err_LR = sum(1-(test_output == test_labels))/n_test;