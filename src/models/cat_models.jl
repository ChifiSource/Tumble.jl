#==
Majority
    Class
        Baseline
==#
@doc """
      Majority class baseline is used to find the most often interpreted
      classification in an array.\n
      --------------------\n
      ==PARAMETERS==\n
     [y] <- Fill with your trainY values. Should be an array of shape (0,1) or (1,0)\n
     --------------------\n
     ==Functions==\n
     predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)\n
     counts() <- Returns a dictionary with the counts of all inserted keys.\n
     highest() <- Will return a Dictionary key with the count as well as the value for the most interpreted classification.
       """
function majClassBaseline(y)
    u=unique(y)
    d=Dict([(i,count(x->x==i,y)) for i in u])
    d = sort(collect(d), by=x->x[2])
    maxkey = d[length(d)]
    predict(xt) = [p = maxkey[1] for p in xt]
    counts() = d
    highest() = maxkey
    (var)->(y;maxkey;d;predict;counts;highest)
end
#==
Logistic
    Regression
==#
@doc """
      Majority class baseline is used to find the most often interpreted
      classification in an array.\n
      --------------------\n
      ==PARAMETERS==\n
     [X] <- Fill with your trainX values. Should be an array of shape (0,1) or (1,0)\n
     [y] <- Fill with your trainy values. Should be an array of shape (0,1) or (1,0)\n
     λ = .0001 <- Lambda Value\n
     fit_intercept = true <- Boolean determines whether to fit an intercept.\n
     max_iter = 1000 <- Determines the maximum number of iterations for the model to perform.\n
     --------------------\n
     ==Functions==\n
     predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)\n
       """
function LogisticRegression(X, y, λ=0.0001, fit_intercept=true, η=0.01, max_iter=1000)
    θ, 𝐉 = logistic_regression_sgd(X, y, 0.0001, true, 0.3, 3000);
    predict(xt) = yhat = predict_class(predict_proba(xt,0))
    cost = 𝐉
    (var) -> (predict;cost)
end
function sigmoid(z)
    return 1 ./ (1 .+ exp.(.-z))
end
function logistic_regression_sgd(X, y, λ, fit_intercept=true, η=0.01, max_iter=1000)

    # Initialize some useful values
    m = length(y); # number of training examples

    if fit_intercept
        # Add a constant of 1s if fit_intercept is specified
        constant = ones(m, 1)
        X = hcat(constant, X)
    else
        X # Assume user added constants
    end

    # Use the number of features to initialise the theta θ vector
    n = size(X)[2]
    θ = zeros(n)

    # Initialise the cost vector based on the number of iterations
    𝐉 = zeros(max_iter)

    for iter in range(1, stop=max_iter)

        # Calcaluate the cost and gradient (∇𝐉) for each iter
        𝐉[iter], ∇𝐉 = regularised_cost(X, y, θ, λ)

        # Update θ using gradients (∇𝐉) for direction and (η) for the magnitude of steps in that direction
        θ = θ - (η * ∇𝐉)
    end

    return (θ, 𝐉)
end
function regularised_cost(X, y, θ, λ)
    m = length(y)
    h = sigmoid(X * θ)
    positive_class_cost = ((-y)' * log.(h))
    negative_class_cost = ((1 .- y)' * log.(1 .- h))
    lambda_regularization = (λ/(2*m) * sum(θ[2 : end] .^ 2))
    𝐉 = (1/m) * (positive_class_cost - negative_class_cost) + lambda_regularization
    ∇𝐉 = (1/m) * (X') * (h-y) + ((1/m) * (λ * θ))
    ∇𝐉[1] = (1/m) * (X[:, 1])' * (h-y)
           return (𝐉, ∇𝐉)
end
function predict_proba(X, θ, fit_intercept=true)
    m = size(X)[1]

    if fit_intercept
        constant = ones(m, 1)
        X = hcat(constant, X)
    else
        X
    end

    h = sigmoid(X * θ)
    return h
end

function predict_class(proba, threshold=0.5)
    return proba .>= threshold
end
