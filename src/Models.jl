#================
Predictive
    Learning
        Models
================#
include("Distributions.jl")
@doc """
      |====== Lathe.models =====\n
      |____________/ Accessories ___________\n
      |_____models.predict(m,xt)\n
      |_____models.Pipeline([steps],model)\n
      |____________/ Continuous models ___________\n
      |_____models.meanBaseline(y)\n
      |_____models.FourSquare(x,y)\n
      |_____models.LinearRegression(x,y)\n
      |_____models.LeastSquare(x,y,Type)\n
      |_____models.PowerLog(prob1,prob2)\n
      |____________/ Categorical Models ___________\n
      |_____models.LogisticRegression(x,y)\n
      |_____models.majBaseline\n
       """
module models
#==
Base
    Models
        Functions
==#
using Lathe
using Random
#===========
Accessories
===========#
@doc """
      Pipelines can contain a predictable Lathe model with preprocessing that
      occurs automatically. This is done by putting X array processing methods
      into the iterable steps, and then putting your Lathe model in.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      model = Lathe.models.meanBaseline(y)\n
      StandardScalar = Lathe.preprocess.StandardScalar\n
      MeanNormalization = Lathe.preprocess.MeanNormalization\n
      steps = [StandardScalar,MeanNormalization]\n
      pipeline = Lathe.models.Pipeline(steps,model)\n
      y_pred = Lathe.models.predict(pipeline,xtrain)\n
      --------------------\n
      HYPER PARAMETERS\n
      steps:: Iterable list (important, use []) of processing methods to be
      performed on the xtrain set. Note that it will not be applied to the
      train set, so preprocessing for the train set should be done before
      model construction.\n
      model:: Takes any Lathe model, uses Lathe.models.predict,\n
      method assersion is still do-able with the dispatch, meaning any model\n
      designed to work with Lathe.models (and Lathe.models.predict) will work\n
      inside of a Lathe pipeline."""
mutable struct Pipeline
    steps
    model
end
function pred_pipeline(m,x)
    x = [x = step(x) for step in m.steps]
    ypr = model.predict(x)
    return(ypr)
end
#==============
========================================================
=======================================================================
            CONTINUOS MODELS               CONTINUOS MODELS
            CONTINUOS MODELS               CONTINUOS MODELS
======================================================================
======================================================================#
#==
Mean
    Baseline
==#
 # Model Type
 @doc """
       A mean baseline is great for getting a basic accuracy score in order
           to make a valid direction for your model.\n
       --------------------\n
       x = [7,6,5,6,5]\n
       y  = [3.4.5.6.3]\n
       xtrain = [7,5,4,5,3,5,7,8]\n
       model = Lathe.models.meanBaseline(y)
       y_pred = model.predict(xtrain)\n
        """
function MeanBaseline(y)
    m = Lathe.stats.mean(m.y)
    predict(xt) =
    xt = [v = m for v in xt]
    (test)->(m;predict)
end

#==
Four
    Square
==#
@doc """
      A FourSquare splits data into four linear least squares, and then
      predicts variables depending on their location in the data (in
      quartile range.) With the corresponding model for said quartile.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      model = Lathe.models.FourSquare(x,y)\n
      yhat = model.predict(xtrain)"""
function FourSquare(x,y)
          # x = q1(r(floor:q1)) |x2 = q2(r(q1:μ)) |x3 = q3(r(q2:q3)) |x4 q4(r(q3:cieling))
          # y' = q1(x * (a / x)) | μ(x * (a / x2)) | q3(x * (a / x3) | q4(x * (a / x4))
              x = m.x
              y
              # Go ahead and throw an error for the wrong input shape:
              xlength = length(x)
              ylength = length(y)
              if xlength != ylength
                  throw(ArgumentError("The array shape does not match!"))
              end
              # Our empty Y prediction list==
              e = []
              # Quad Splitting the data ---->
              # Split the Y
              y2,range1 = Lathe.preprocess.SortSplit(y)
              y3,range2 = Lathe.preprocess.SortSplit(y2)
              y4,range3 = Lathe.preprocess.SortSplit(y3)
              y5,range4 = Lathe.preprocess.SortSplit(y4)
              yrange5 = y5
              # Split the x train
              x1,xrange1 = Lathe.preprocess.SortSplit(x)
              x2,xrange2 = Lathe.preprocess.SortSplit(x1)
              x3,xrange3 = Lathe.preprocess.SortSplit(x2)
              x4,xrange4 = Lathe.preprocess.SortSplit(x3)
              xrange5 = y5
              # Fitting the 4 linear regression models ---->
              regone = LeastSquare(xrange1,range1,:LIN)
              regtwo = LeastSquare(xrange2,range2,:LIN)
              regthree = LeastSquare(xrange3,range3,:LIN)
              regfour = LeastSquare(xrange4,range4,:LIN)
              regfive = LeastSquare(xrange5,yrange5,:LIN)
              # Split the train Data
              xt1,xtrange1 = Lathe.preprocess.SortSplit(xt)
              xt2,xtrange2 = Lathe.preprocess.SortSplit(xt1)
              xt3,xtrange3 = Lathe.preprocess.SortSplit(xt2)
              xt4,xtrange4 = Lathe.preprocess.SortSplit(xt3)
              xtrange5 = xt4
              # Get min-max
              xtrange1min = minimum(xtrange1)
              xtrange1max = maximum(xtrange1)
              xtrange2min = minimum(xtrange2)
              xtrange2max = maximum(xtrange2)
              xtrange3min = minimum(xtrange3)
              xtrange3max = maximum(xtrange3)
              xtrange4min = minimum(xtrange4)
              xtrange4max = maximum(xtrange4)
              xtrange5min = minimum(xtrange5)
              # Ranges for ifs
              condrange1 = (xtrange1min:xtrange1max)
              condrange2 = (xtrange2min:xtrange2max)
              condrange3 = (xtrange3min:xtrange3max)
              condrange4 = (xtrange4min:xtrange4max)
              # This for loop is where the dimension's are actually used:
              for i in xt
                  if i in condrange1
                      ypred = predict(regone,i)
                  elseif i in condrange2
                      ypred = predict(regtwo,i)
                  elseif i in condrange3
                      ypred = predict(regthree,i)
                  elseif i in condrange4
                      ypred = predict(regfour,i)
                  else
                      ypred = predict(regfive,i)
                  end
                  append!(e,ypred)
              end
              return(e)
      end
#==
Linear
    Regression
==#

@doc """
      Linear Regression is a well-known linear function used for predicting
      continuous features with a mostly linear or semi-linear slope.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      model = Lathe.models.LinearRegression(x,y)
      y_pred = model.predict(xtrain)\n
       """
function LinearRegression(x,y)
    # a = ((∑y)(∑x^2)-(∑x)(∑xy)) / (n(∑x^2) - (∑x)^2)
    # b = (x(∑xy) - (∑x)(∑y)) / n(∑x^2) - (∑x)^2
    if length(x) != length(y)
        throw(ArgumentError("The array shape does not match!"))
    end
    # Get our Summations:
    Σx = sum(x)
    Σy = sum(y)
    # dot x and y
    xy = x .* y
    # ∑dot x and y
    Σxy = sum(xy)
    # dotsquare x
    x2 = x .^ 2
    # ∑ dotsquare x
    Σx2 = sum(x2)
    # n = sample size
    n = length(x)
    # Calculate a
    a = (((Σy) * (Σx2)) - ((Σx * (Σxy)))) / ((n * (Σx2))-(Σx^2))
    # Calculate b
    b = ((n*(Σxy)) - (Σx * Σy)) / ((n * (Σx2)) - (Σx ^ 2))
    # The part that is super struct:
    predict(xt) = (xt = [i = a + (b * i) for i in xt])
    (test)->(a;b;predict)
end
#==
Linear
    Least
     Square
==#
@doc """
      Least Squares is ideal for predicting continous features.
      Many models use Least Squares as a base to build off of.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      Type = :LIN\n
      model = Lathe.models.LeastSquare(x,y,Type)\n
      y_pred = Lathe.models.predict(model,xtrain)\n
      -------------------\n
      HYPER PARAMETERS\n
      Type:: Type determines which Linear Least Square algorithm to use,
      :LIN, :OLS, :WLS, and :GLS are the three options.\n
      - :LIN = Linear Least Square Regression\n
      - :OLS = Ordinary Least Squares\n
      - :WLS = Weighted Least Squares\n
      - :GLS = General Least Squares
       """
function LeastSquare(x,y,Type)
    if length(x) != length(y)
        throw(ArgumentError("The array shape does not match!"))
    end
    if Type == :LIN
        xy = x .* y
        sxy = sum(xy)
        n = length(x)
        x2 = x .^ 2
        sx2 = sum(x2)
        sx = sum(x)
        sy = sum(y)
        # Calculate the slope:
        slope = ((n*sxy) - (sx * sy)) / ((n * sx2) - (sx)^2)
     # Calculate the y intercept
        b = (sy - (slope*sx)) / n
    end
    predict(xt) =
    if Type == :LIN
        (xt = [z = (slope * x) + b for x in xt])
    end
    (test)->(slope;b;predict)
end
#==
Ridge
    Regression
==#
@doc """
      Ridge Regression is another regressor ideal for predicting linear,
          continuous features.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.OneHotEncode(array)\n
       """
function RidgeRegression(x,y)

end

#======================================================================
=======================================================================
            CATEGORICAL MODELS             CATEGORICAL MODELS
            CATEGORICAL MODELS             CATEGORICAL MODELS
======================================================================
======================================================================#
#==
Majority
    Class
        Baseline
==#
@doc """
      FUNCTION NOT YET WRITTEN\n
      Majority class baseline is used to find the most often interpreted
      classification in an array.\n
      --------------------\n
       """
function MajBaseline

end
#==
Power Log
==#
function PowerLog(p1::Float64,p2::Float64; alpha::Float64 = 0.05, rsq::Real = 0)
    pd = p2 - p1
    l1 = p1/(1-p1)
    l2 = p2/(1-p2)
    θ = l2 / l1
    or = θ
    λ = log(θ)
    λ2 = λ^2
    za = quantile(normal(),1-alpha)
    println("One-tailed test: alpha = ",alpha,", p1 = ",p1,", p2 = ",p2,", rsq = ",rsq,", odds ratio = ",or)
    δ = (1 + (1 + λ2)*exp(5 * λ2/4))/(1 + exp(-1*λ2/4))
    pwr = zeros(Float64,8)
    nn = zeros(Int64,8)
    i = 1
    for power = 0.6:.05:.95
        zb = quantile(normal(),power)

        N = ((za + zb*exp(-1 * λ2/4))^2 * (1 + 2*p1*δ))/(p1*λ2)
        N /= (1 - rsq)
        pwr[i] = power
        nn[i] = ceil(Int64,N)
        i += 1
    end
    (var) -> (pwr)
end
#==
Multinomial
    Naive
        Bayes
==#
function MultinomialNB(x,y)

end
#==
Logistic
    Regression
==#
#==
@doc """
      One hot encoder replaces a single feature with sub arrays containing
      boolean values (1 or 0) for each individual category.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.OneHotEncode(array)\n
       """ ->
       ==#
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
#==
Nueral
    Network
        Framework
==#
function calculate_activation_forward(A_pre , W , b , function_type)
    if (function_type == "sigmoid")
        Z , linear_step_cache = forward_linear(A_pre , W , b)
        A , activation_step_cache = sigmoid(Z)
    elseif (function_type == "relu")
        Z , linear_step_cache = forward_linear(A_pre , W , b)
        A , activation_step_cache = relu(Z)
    end
    cache = (linear_step_cache , activation_step_cache)
    return A , cache
end

function model_forward_step(X , params)
    all_caches = []
    A = X
    L = length(params)/2
    for l=1:L-1
        A_pre = A
        A , cache = calculate_activation_forward(A_pre , params[string("W_" , string(Int(l)))] , params[string("b_" , string(Int(l)))] , "relu")
        push!(all_caches , cache)
    end
    A_l , cache = calculate_activation_forward(A , params[string("W_" , string(Int(L)))] , params[string("b_" , string(Int(L)))] , "sigmoid")
    push!(all_caches , cache)
    return A_l , all_caches
end
function cost_function(AL , Y)
    cost = -mean(Y.*log.(AL) + (1 .- Y).*log.(1 .- AL))
    return cost
end
function forward_linear(A , w , b)
    Z = w*A .+ b
    cache = (A , w , b)
    return Z,cache
end
function init_param(layer_dimensions)
    param = Dict()
    for l=0:length(layer_dimensions)
        param[string("W_" , string(l-1))] = rand(layer_dimensions[l] ,
				layer_dimensions[l-1])*0.1
        param[string("b_" , string(l-1))] = zeros(layer_dimensions[l] , 1)
    end
    return param
end

function backward_linear_step(dZ , cache)
    A_prev , W , b = cache
    m = size(A_prev)[2]
    dW = dZ * (A_prev')/m
    db = sum(dZ , dims = 2)/m
    dA_prev = (W')* dZ
    return dW , db , dA_prev
end
function relu(X)
    rel = max.(0,X)
    return rel , X
end
function backward_relu(dA , cache_activation)
    return dA.*(cache_activation.>0)
end
function backward_sigmoid(dA , cache_activation)
    return dA.*(sigmoid(cache_activation)[1].*(1 .- sigmoid(cache_activation)[1]))
end
function backward_activation_step(dA , cache , activation)
    linear_cache , cache_activation = cache
    if (activation == "relu")
        dZ = backward_relu(dA , cache_activation)
        dW , db , dA_prev = backward_linear_step(dZ , linear_cache)
    elseif (activation == "sigmoid")
        dZ = backward_sigmoid(dA , cache_activation)
        dW , db , dA_prev = backward_linear_step(dZ , linear_cache)
    end
    return dW , db , dA_prev

end
function model_backwards_step(A_l , Y , caches)
    grads = Dict()
    L = length(caches)
    m = size(A_l)[2]
    Y = reshape(Y , size(A_l))
    dA_l = (-(Y./A_l) .+ ((1 .- Y)./( 1 .- A_l)))
    current_cache = caches[L]
    grads[string("dW_" , string(L))] , grads[string("db_" , string(L))] , grads[string("dA_" , string(L-1))] = backward_activation_step(dA_l , current_cache , "sigmoid")
    for l=reverse(0:L-2)
        current_cache = caches[l+1]
        grads[string("dW_" , string(l+1))] , grads[string("db_" , string(l+1))] , grads[string("dA_" , string(l))] = backward_activation_step(grads[string("dA_" , string(l+1))] , current_cache , "relu")

    end
    return grads
end

function update_param(parameters , grads , learning_rate)
    L = Int(length(parameters)/2)
    for l=0:(L-1)
        parameters[string("W_" , string(l+1))] -= learning_rate.*grads[string("dW_" , string(l+1))]
        parameters[string("b_",string(l+1))] -= learning_rate.*grads[string("db_",string(l+1))]
    end
    return parameters
end
@doc """
      'Network' specifically constructs a convolutional nueral network,\n
      though this network is still being worked on in both methodology\n
      and in a physical manner,\n thanks for understanding.
       """
function Network(X,Y,layers_dimensions,n_iter)
    params = init_param(layers_dimensions)
    costs = []
    iters = []
    accuracy = []
    for i=1:n_iter
        A_l , caches  = model_forward_step(X , params)
        cost = cost_function(A_l , Y)
        acc = check_accuracy(A_l , Y)
        grads  = model_backwards_step(A_l , Y , caches)
        params = update_param(params , grads , learning_rate)
        println("Iteration ->" , i)
        println("Cost ->" , cost)
        println("Accuracy -> " , acc)
        push!(iters , i)
        push!(costs , cost)
        push!(accuracy , acc)
    end
    predict(xt) = (xt + 4)
    (test)->(costs;params;predict)
end

#
predict(m::Pipeline,x) = pred_pipeline(m,x)
#----------------------------------------------
end
