#==
Linear
    Regression
==#
@doc """
      Linear Regression is a well-known linear function used for predicting
      continuous features with a mostly linear or semi-linear slope.\n
      --------------------\n
      ==PARAMETERS==\n
     [y] <- Fill with your trainY values. Should be an array of shape (0,1) or (1,0)\n
     [x] <- Fill in with your trainX values. Should be an array of shape (0,1) or (1,0)\n
     --------------------\n
     ==Functions==\n
     predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)
       """
mutable struct LinearRegression{P} <: LinearModel
    a::Float64
    b::Float64
    predict::P
    regressors::Array{LinearModel}
    function LinearRegression(x::Array, y::Array; cuda = false)
        # a = ((∑y)(∑x^2)-(∑x)(∑xy)) / (n(∑x^2) - (∑x)^2)
        # b = (x(∑xy) - (∑x)(∑y)) / n(∑x^2) - (∑x)^2
        regressors = []
        vals = cudacheck([Array(x), Array(y)], cuda)
        x, y = vals[1], vals[2]
        checkdims(x, y)
        xy, x2 = x .* y, x .^ 2
        Σx, Σy, Σxy, Σx2  = sum(x), sum(y), sum(xy), sum(x2)
        n = length(x)
        a = (((Σy) * (Σx2)) - ((Σx * (Σxy)))) / ((n * (Σx2)) - (Σx ^ 2))
        b = ((n * (Σxy)) - (Σx * Σy)) / ((n * (Σx2)) - (Σx ^ 2))
        predict(xt::Array) = (xt = [i = a + (b * i) for i in xt])
        P = typeof(predict)
        return new{P}(a, b, predict, [])
    end
        function LinearRegression(x::DataFrame, y::Array; cuda = false)
            regressors = [LinearRegression(Array(feature),
             y) for feature in eachcol(x)]
            a = nothing
            b = nothing
            for m in regressors
                if a != nothing
                    a = mean(a, m.a)
                    b = mean(b, m.b)
                else
                    a = m.a
                    b = m.b
                end
            end
            predict(xt::DataFrame) = _compare_predCon(models, xt)
            P = typeof(predict)
            return new{P}(a, b, predict, regressors)
    end
end
#==
Linear
    Least
     Square
==#
"""

      Least Square regressors are ideal for predicting continous features.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      Type = :LIN\n
      model = models.LeastSquare(x,y,Type)\n
      y_pred = models.predict(model,xtrain)\n
      -------------------\n
      HYPER PARAMETERS\n
      --------------------\n
      ==Functions==\n
      predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)
       """
mutable struct LinearLeastSquare{P} <: LinearModel
    a::Float64
    b::Float64
    predict::P
    regressors::Array{LinearModel}
    function LinearLeastSquare(x::AbstractArray, y::AbstractArray)
        checkdims(Array(x), Array(y))
        xy, x2 = x .* y, x .^ 2
        Σxy, Σx2, Σx, Σy = sum(xy), sum(x2), sum(x), sum(y)
        n = length(x)
        a = ((n * Σxy) - (Σx * Σy)) / ((n * Σx2) - (Σx) ^ 2)
        b = (Σy - (a * Σx)) / n
        predict(xt) = [z = (a * x) + b for x in xt]
        new{typeof(predict)}(a, b, predict, [])
    end
    function LinearLeastSquare(x::DataFrame, y::AbstractArray, cuda = false)
        vals = cudacheck([x, y], cuda)
        x, y = Array(vals[1]), Array(vals[2])
        regressors = [LinearLeastSquare(Array(feature),
         y) for feature in eachcol(x)]
        a = nothing
        b = nothing
        for m in regressors
            if a != nothing
                a = mean(a, m.a)
                b = mean(b, m.b)
            else
                a = m.a
                b = m.b
            end
        end
        predict(xt::DataFrame) = _compare_predCon(models, xt)
        P = typeof(predict)
        new{typeof(predict)}(a, b, predict, regressors)
end
end
#==
Lasso
    Regression
==#
function lasso_cost(X, Y, B, alp)
     m = length(Y)
     cost = (sum(((X .* B) - Y).^2)/2) +(sum(broadcast(abs, B)))*alp/2
     return cost
end
function gradientDescent(X, Y, B, learningRate, numIterations, alp)
    m = length(Y)
    for iteration in 1:numIterations
        loss = (X .* B) - Y
        for i in 1:length(X)
            if(B[i]==0)
                B[i]=1
            end
        end
        gradient = ((X' * loss) .+ (alp .* (broadcast(abs, B) ./ B))) / m
        B = B - learningRate * gradient
    end
    return B
end
mutable struct LassoRegression{P}
    x::Array
    y::Array
    predict::P
    lambda::Float64
    i::Float64
    LearningRate::Float64
    n_iterations::Int64
    function LassoRegression(x, y, lambda = 0; lr = .01, n_iterations = 1200)
        i = 10 ^ 6
        predict(xt) = pred_lasso(x, y, xt, i, learningRate = lr, n_iterations = n_iterations)
        new{typeof(predict)}(x, y, predict, lambda, i, lr, n_iterations)
    end
    function pred_lasso(x, y, B, i; learningRate = .01, n_iterations = 1200)
        costm = 10 ^ 30
        newB = zeros(size(B))
          while i > 0.001
             B1 = gradientDescent(x, y, B, learningRate, 1200,i)
             cost = lasso_cost(x, y, B1, i)
            if (cost < costm)
                costm = cost
                newB = B1
                lambda = i
            end
         i=i/2
        end
        return(B .* newB)
    end
end
