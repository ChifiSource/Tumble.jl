
mutable struct SVD{P} <: Transformer
    predict::P
    iter::Int64
    function SVD(maxiter=1000)
        predict(A, r) = -svd(A, r, maxiter = iter)
        iter = maxiter
        new{typeof(predict)}(predict, iter)
    end
    function _svd(A, r; iter = 100)
        V = randn(size(A,2),r) # random initialization
        for _ = 1:maxiter
            W = A * V
            Z = A' * W
            V, R = mmids_gramschmidt(Z)
        end
        W = A * V
        S = [norm(W[:, i]) for i=1:size(W,2)] # singular values
        U = reduce(hcat,[W[:,i]/S[i] for i=1:size(W,2)]) # left singular vectors
        return U * S * V'
end
end
