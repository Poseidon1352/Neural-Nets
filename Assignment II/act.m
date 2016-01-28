function [Z,H_] = act(A,str)
    switch str
        case 'tanh'
            Z = tanh(A);
            H_ = 1 - Z.^2;
        case 'sigmoid'
            Z = 1./(1 + exp(-A));
            H_ = Z.*(1-Z);
        otherwise
            Z = (A>0).*A;
            H_ = A > 0 + 0.5*(A==0);
    end
end