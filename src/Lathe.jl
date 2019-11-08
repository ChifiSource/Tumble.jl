#===============================
<-----------Lathe.jl----------->
Programmed by Emmett Boudreau
    <emmett@emmettboudreau.com>
        <http://emmettboudreau.com>
MIT General Open Source License
    (V 3.0.0)
        Free for Modification and
        Redistribution
Thank you for your forks!
<-----------Lathe.jl----------->
38d8eb38-e7b1-11e9-0012-376b6c802672
#[deps]
DataFrames.jl
Random.jl
JLD2
FileIO
================================#
module Lathe
using FileIO
using JLD2
using DataFrames
using Random
#================
Stats
    Module
================#
module stats
#<----Mean---->
function mean(array)
    observations = length(array)
    average = sum(array)/observations
    return(average)
end
#<----Mode---->
function mode(array)
    m = findmax(array)
    return(m)
end
#<----Variance---->
function variance(array)
    me = mean(array)
    sq = sum(array) - me
    squared_mean = sq ^ 2
    return(squared_mean)
end
#<----Confidence Intervals---->
function confiints(data, confidence=.95)
    mean = mean(data)
    std = std(data)
    stderr = standarderror(data)
#    interval = stderr * scs.t.ppf((1 + confidence) / 2.0, n-1)
#    return (mean-interval, mean+interval)
end
#<----Standard Error---->
function standarderror(data)
    std = std(data)
    sample = length(data)
    ste = (std/sqrt(sample))
    return(ste)
end
#<----Standard Deviation---->
function std(array3)
    m = mean(array3)
    [i = (i-m) ^ 2 for i in array3]
    m = mean(array3)
    m = sqrt(m)
    return(m)
end
#<---- Correlation Coefficient --->
function correlationcoeff(x,y)
    n = length(x)
    yl = length(y)
    if n != yl
        throw(ArgumentError("The array shape does not match!"))
    end
    xy = x .* y
    sx = sum(x)
    sy = sum(y)
    sxy = sum(xy)
    x2 = x .^ 2
    y2 = y .^ 2
    sx2 = sum(x2)
    sy2 = sum(y2)
    r = ((n*sxy) - (sx * sy)) / (sqrt((((n*sx2)-(sx^2)) * ((n*sy2)-(sy^2)))))
    return(r)
end
#<----Z score---->
function z(array)

end
#<----Quartiles---->
# - First
function firstquar(array)
    m = median(array)
    q15 = array / m
    q1 = array / m
    return(q)
end
# - Second(median)
function secondquar(array)
    m = median(array)
    return(m)
end
# - Third
function thirdquar(array)
    q = median(array)
    q = q * 1.5
end
# <---- Rank ---->
function getranks(array,rev = false)
    sortedar = sort!(array,rev=rev)
    num = 1
    list = []
    for i in sortedar
        append!(list,i)
        num = num + 1
    end
    return(list)
end
#-------Inferential-----------__________
#<----Inferential Summary---->
function inf_sum(data,grdata)
    #Doing our calculations
    t = independent_t(data,grdata)
    f = f_test(data,grdata)
#    low,high = confiints(data)
    var = variance(data)
    grvar = variance(grdata)
    avg = mean(data)
    gravg = mean(grdata)
    sampstd = std(data)
    grstd = std(grdata)
    #Printing them out
    println("================")
    println("     Lathe.stats Inferential Summary")
    println("     _______________________________")
    println("N: ",length(grdata))
    println("x̅: ",avg)
    println("μ: ",gravg)
    println("s: ",sampstd)
    println("σ: ",grstd)
    println("var(x): ",var)
    println("σ2: ",grvar)
#    println("Low Confidence interval: ",low)
#    println("High Confidence interval: ",high)
    println("α ",t)
    println("Fp: ",f)
    println("================")
end
#<----T Test---->
# - Independent
function independent_t(sample,general)
    sampmean = mean(sample)
    genmean = mean(general)
    samples = length(sample)
    m = genmean
    [i = (i-m) ^ 2 for i in general]
    m = mean(general)
    m = sqrt(m)
    std = m
    t = (sampmean - genmean) / (std / sqrt(samples))
    return(t)
end
# - Paired
function paired_t(var1,var2)

end
#<---- Correlations ---->
# - Spearman
function spearman(var1,var2)

end
# - Pearson
function pearson(x,y)
    sx = std(x)
    sy = std(y)
    x̄ = mean(x)
    ȳ = mean(x)
    [i = (i-x̄) / sx for i in x]
    [i = (i-ȳ) / sy for i in y]
    n1 = n-1
    mult = x .* y
    sq = sum(mult)
    corrcoff = sq / n1
    return(corrcoff)
end
# <---- Chi Distribution --->
function chidist(x,e)

end
#<---- Chi-Square ---->
function chisq(var1,var2)

end
#<---- ANOVA ---->
function anova(var1,var2)

end
#<---- Wilcoxon ---->
# - Wilcoxon Rank-Sum Test
function wilcoxrs(var1,var2)

end
function wilcoxsr(var1,var2)

end
#<---- Binomial Distribution ---->
function binomialdist(positives,size)
    # p = n! / x!(n-x!)*π^x*(1-π)^N-x
    n = size
    x = positives
    factn = factorial(n)
    factx = factorial(x)
    p = factn / factx * (n-factx) * π ^ x * (1-π)^n - x
    println("P - ",p)
    pxr = factn / (factx * (n-x)) * p^p * (1-p)^(n-x)
    return(pxr)
end
#<---- Sign Test ---->
function sign(var1,var2)
    sets = var1 .- var2
    positives = []
    negatives = []
    zeros = []
    for i in sets
        if i == 0
            zeros.append(i)
        elseif i > 0
            positives.append(i)
        elseif i < 0
            negatives.append(i)
        end
    end
    totalpos = length(positives)
    totallen = length(sets)
    ans = binomialdist(positives,totallen)
    return(ans)
end
#<---- F-Test---->
function f_test(sample,general)
    totvariance = variance(general)
    sampvar = variance(sample)
    f =  sampvar / totvariance
    return(f)
end
#-------Bayesian--------------___________
#<----Bayes Theorem---->
#P = prob, A = prior, B = Evidence,
function bay_ther(p,a,b)
    psterior = (p*(b|a) * p*(a)) / (p*b)
    return(psterior)
end
function cond_prob(p,a,b)
    psterior = bay_ther(p,a,b)
    cond = p*(a|b)
    return(cond)
end

#---------------------------
end
#================
Model
    Validation
        Module
================#
module validate
#-------Model Metrics--------____________
using Lathe
## <---- Mean Absolute Error ---->
function mae(actual,pred)
    l = length(actual)
    lp = length(pred)
    if l != lp
        throw(ArgumentError("The array shape does not match!"))
    end
    result = actual-pred
    maeunf = Lathe.stats.mean(result)
    if maeunf < 0
        maeunf = (maeunf - maeunf) - maeunf
    end
    return(maeunf)
end
# <---- R Squared ---->
function r2(actual,pred)
    l = length(actual)
    lp = length(pred)
    if l != lp
        throw(ArgumentError("The array shape does not match!"))
    end
    r = Lathe.stats.correlationcoeff(actual,pred)
    rsq = r^2
    rsq = rsq * 100
    return(rsq)
end
# <---- Mean Squared Error ---->
function mse(actual,pred)
    l = length(actual)
    lp = length(pred)
    if l != lp
        throw(ArgumentError("The array shape does not match!"))
    end
    result = actual-pred
    result = result .^ 2
    maeunf = Lathe.stats.mean(result)
    return(maeunf)
end
# <---- Binomial Accuracy ---->
function binomialaccuracy(actual,pred)
    # p = n! / x!(n-x!)*π^x*(1-π)^N-x
end
# --- Get Permutation ---
function getPermutation(model)

end
#--------------------------------------------
# End
end
#================
Preprocessing
     Module
================#
module preprocess
using Random
using Lathe
#===============
Generalized
    Data
        Processing
===============#
# Train-Test-Split-----
function TrainTestSplit(df,at = 0.75)
    sample = randsubseq(1:size(df,1), at)
    trainingset = df[sample, :]
    notsample = [i for i in 1:size(df,1) if isempty(searchsorted(sample, i))]
    testset = df[notsample, :]
    return(trainingset,testset)
end
# Array-Split ----------
function ArraySplit(data, at = 0.7)
    n = length(data)
    idx = Random.shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
    return(test_idx,train_idx)
end
# Sort-Split -------------
function SortSplit(data, at = 0.25, rev=false)
  n = length(data)
  sort!(data, rev=rev)  # Sort in-place
  train_idx = view(data, 1:floor(Int, at*n))
  test_idx = view(data, (floor(Int, at*n)+1):n)
  return(test_idx,train_idx)
end
# Unshuffled Split ----
function Uniform_Split(data, at = 0.7)
    n = length(data)
    idx = data
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
    return(test_idx,train_idx)
end
#=======
Numerical
    Scaling
=======#
# ---- Rescalar (Standard Deviation) ---
function Rescalar(array)
    v = []
    min = minimum(array)
    max = maximum(array)
    for i in array
        x = (i-min) / (max - min)
        append!(v,x)
    end
    return(v)
end
# ---- Arbitrary Rescalar ----
function ArbitraryRescale(array)
    v = []
    a = minimum(array)
    b = maximum(array)
    for i in array
        x = a + ((i-a*i)*(b-a)) / (b-a)
        append!(v,x)
    end
    return(v)
end
# ---- Mean Normalization ----
function MeanNormalization(array)
    avg = Lathe.stats.mean(array)
    v = []
    a = minimum(array)
    b = maximum(array)
    for i in array
        m = (i-avg) / (b-a)
        append!(v,m)
    end
end
# ---- Z Normalization ----
function StandardScalar(array)
    q = Lathe.stats.std(array)
    avg = Lathe.stats.mean(array)
    v = []
    for i in array
        y = (i-avg) / q
        append!(v,y)
    end
    return(v)
end
# ---- Unit L-Scale normalize ----
function Unit_LScale(array)

end
#==========
Categorical
    Encoding
==========#
# <---- One Hot Encoder ---->
function OneHotEncode(array)

end
#-----------------------------
end
#================
Predictive
    Learning
        Models
================#
module models
#==
Baseline
    Model
==#
using Lathe
using Random
#Show models shows all the models that are stable
#And ready for use in the library
function showmodels()
    println("    Lathe.JL    ")
    println("________________")
    println("Current")
    println("    Usable")
    println("       Models")
    println("================")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Continuous Models")
    println("-----------------")
    println("_____Linear_____")
    println("meanBaseline(y)")
    println("LinearLeastSquare(x,y,Type)")
    println("LinearRegression(x,y)")
    println("-----------------")
    println("___Non-Linear___")
    println("FourSquare(x,y)")
    println("ExponentialScalar(x,y)")
end
#Takes model, and X to predict, and returns a y prediction
function predict(m,x)
    if typeof(m) == FourSquare
        y_pred = pred_foursquare(m,x)
    end
    if typeof(m) == majBaseline
        y_pred = pred_catbaseline(m,x)
    end
    if typeof(m) == RegressionTree
        y_pred = pred_regressiontree(m,x)
    end
    if typeof(m) == LinearRegression
        y_pred = pred_LinearRegression(m,x)
    end
    if typeof(m) == meanBaseline
        y_pred = pred_meanbaseline(m,x)
    end
    if typeof(m) == RidgeRegression
        y_pred = pred_ridgeregression(m,x)
    end
    if typeof(m) == LinearLeastSquare
        y_pred = pred_linearleastsquare(m,x)
    end
    if typeof(m) == LogisticRegression
        y_pred = pred_logisticregression(m,x)
    end
    if typeof(m) == ExponentialScalar
        y_pred = pred_exponentialscalar(m,x)
    end
    if typeof(m) == MultipleLinearRegression
        y_pred = pred_multiplelinearregression(m,x)
    end
    return(y_pred)
end
# The help function:
function help(args)
    if typeof(args) == LogisticRegression
        println("Logistic")
    elseif typeof(args) == meanBaseline
        println("MeanBaseline")
    end
end
#======================================================================
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
mutable struct meanBaseline
    y
end
#----  Callback
function pred_meanbaseline(m,xt)
    e = []
    m = Lathe.stats.mean(m.y)
    print("-Lathe.models Mean Baseline-")
    print("mean: ",m)
    for i in xt
        append!(e,m)
    end
    return(e)
end
#==
Multi-
    Gap
 - A quad range predictor, on steroids. -
==#
# Model Type
mutable struct RegressionTree
    x
    y
    n_divisions
    divisionsize
end
#----  Callback
# WIP <TODO>
function pred_regressiontree(m,xt)
        x = m.x
        y = m.y
        xtcopy = xt
        divs = m.n_divisions
        size = m.divisionsize
        # Go ahead and throw an error for the wrong input shape:
        xlength = length(x)
        ylength = length(y)
        if xlength != ylength
            throw(ArgumentError("The array shape does not match!"))
        end
        # Now we also need an error for when the total output of the
        #    division size and n divisions is > 100 percent
        divisions = size * divs
        if divisions != 1
            throw(ArgumentError("Invalid hyperparameters!: divisions * number of
            divisions must be = to 100 percent!"))
        end
        # Empty list
        e = []
        while divs > 0
            predictorx,x = Lathe.preprocess.SortSplit(x,size)
            predictory,y = Lathe.preprocess.SortSplit(y,size)
            predictorxt,xtcopy = Lathe.preprocess.SortSplit(xtcopy,size)
            currentrange = (minimum(predictorxt):maximum(predictorxt))
            linregmod = LinearRegression(predictorx,predictory)
            # Recursion replacement method:
            [predict(LinearRegression(predictorx,
            predictory),x) for x in currentrange]
            divs = divs - 1
        end
        return(xt)
end
#==
Four
    Square
==#
# Model Type
mutable struct FourSquare
    x
    y
end
#----  Callback
function pred_foursquare(m,xt)
    # x = q1(r(floor:q1)) |x2 = q2(r(q1:μ)) |x3 = q3(r(q2:q3)) |x4 q4(r(q3:cieling))
    # y' = q1(x * (a / x)) | μ(x * (a / x2)) | q3(x * (a / x3) | q4(x * (a / x4))
        x = m.x
        y = m.y
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
        regone = LinearLeastSquare(xrange1,range1)
        regtwo = LinearLeastSquare(xrange2,range2)
        regthree = LinearLeastSquare(xrange3,range3)
        regfour = LinearLeastSquare(xrange4,range4)
        regfive = LinearLeastSquare(xrange5,yrange5)
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
Isotonic
    Regression
==#
mutable struct IsotonicRegression
    x
    y
end
function pred_isotonicregression(m,xt)
    if length(m.x) != length(m.y)
        throw(ArgumentError("The array shape does not match!"))
    end
end
#==
Multiple
    Linear
        Regression
==#
mutable struct MultipleLinearRegression
    x
    y
end
function pred_multiplelinearregression(m,xt)
    if length(m.x) != length(xt)
        throw(ArgumentError("Bad Feature Shape |
        Training Features are not equal!",))
    end
    y_pred = []
    y = m.y
    x = m.x
    for z in xt
        m = LinearRegression(z,y)
        for b in z
            predavg = []
            for i in x
                for b in i
                    pred = predict(m,b)
                    append!(predavg,pred)
                end
            end
        mn = Lathe.stats.mean(predavg)
    end
        append!(y_pred,mn)
    end
    return(y_pred)
end
#==
Linear
    Regression
==#
mutable struct LinearRegression
    x
    y
end
#----  Callback
function pred_LinearRegression(m,xt)
    # a = ((∑y)(∑x^2)-(∑x)(∑xy)) / (n(∑x^2) - (∑x)^2)
    # b = (x(∑xy) - (∑x)(∑y)) / n(∑x^2) - (∑x)^2
    if length(m.x) != length(m.y)
        throw(ArgumentError("The array shape does not match!"))
    end
    # Get our x and y as easier variables
    x = m.x
    y = m.y
    # Get our Summatations:
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
    [i = a+(b*i) for i in xt]
    y_pred = []
    for i in xt
        z = a+(b*i)
        append!(y_pred,z)
    end
    return(y_pred)
end
#==
Linear
    Least
     Square
==#
mutable struct LinearLeastSquare
    x
    y
    Type
end
function pred_linearleastsquare(m,xt)
    if length(m.x) != length(m.y)
        throw(ArgumentError("The array shape does not match!"))
    end
    if m.Type == :REG
        x = m.x
        y = m.y
        # Summatation of x*y
        xy = x .* y
        sxy = sum(xy)
        # N
        n = length(x)
        # Summatation of x^2
        x2 = x .^ 2
        sx2 = sum(x2)
        # Summatation of x and y
        sx = sum(x)
        sy = sum(y)
        # Calculate the slope:
        slope = ((n*sxy) - (sx * sy)) / ((n * sx2) - (sx)^2)
        # Calculate the y intercept
        b = (sy - (slope*sx)) / n
        # Empty prediction list:
        y_pred = []
        for i in xt
            pred = (slope*i)+b
            append!(y_pred,pred)
        end
    end
    if m.Type == :OLS

    end
    if m.Type == :WLS

    end
    if m.Type == :GLS

    end
    if m.Type == :GRG

    end
    return(y_pred)
end
#==
Ridge
    Regression
==#
mutable struct RidgeRegression
    x
    y
end
function pred_ridgeregression(m,xt)
    if length(m.x) != length(m.y)
        throw(ArgumentError("The array shape does not match!"))
    end
end
#==
Logistic
    Regression
==#
mutable struct LogisticRegression
    x
    y
end
function pred_logisticregression(m,xt)
    if length(m.x) != length(m.y)
        throw(ArgumentError("The array shape does not match!"))
    end

end
#==
Binomial
    Distribution
==#
mutable struct BinomialDistribution
    x
    y
end
function pred_binomialdist(m,xt)
    if length(m.x) != length(m.y)
        throw(ArgumentError("The array shape does not match!"))
    end
end
#==
Exponential
    Scalar
==#
mutable struct ExponentialScalar
    x
    y
end
function pred_exponentialscalar(m,xt)
    x = m.x
    y = m.y
    at = 0.25
    xdiv1,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv2,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv3,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv4,x = Lathe.preprocess.SortSplit(x,.05)
    ydiv1,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv2,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv3,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv4,y = Lathe.preprocess.SortSplit(y,.05)
    scalarlist1 = ydiv1 ./ xdiv1
    scalarlist2 = ydiv2 ./ xdiv2
    scalarlist3 = ydiv3 ./ xdiv3
    scalarlist4 = ydiv3 ./ xdiv3
    xdiv1,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv2,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv3,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv4,x = Lathe.preprocess.SortSplit(x,.05)
    ydiv1,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv2,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv3,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv4,y = Lathe.preprocess.SortSplit(y,.05)
    scalarlist6 = ydiv1 ./ xdiv1
    scalarlist7 = ydiv2 ./ xdiv2
    scalarlist8 = ydiv3 ./ xdiv3
    scalarlist9 = ydiv3 ./ xdiv3
    xdiv1,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv2,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv3,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv4,x = Lathe.preprocess.SortSplit(x,.05)
    ydiv1,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv2,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv3,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv4,y = Lathe.preprocess.SortSplit(y,.05)
    scalarlist10 = ydiv1 ./ xdiv1
    scalarlist11 = ydiv2 ./ xdiv2
    scalarlist12 = ydiv3 ./ xdiv3
    scalarlist13 = ydiv3 ./ xdiv3
    xdiv1,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv2,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv3,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv4,x = Lathe.preprocess.SortSplit(x,.05)
    ydiv1,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv2,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv3,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv4,y = Lathe.preprocess.SortSplit(y,.05)
    scalarlist14 = ydiv1 ./ xdiv1
    scalarlist15 = ydiv2 ./ xdiv2
    scalarlist16 = ydiv3 ./ xdiv3
    scalarlist17 = ydiv3 ./ xdiv3
    xdiv1,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv2,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv3,x = Lathe.preprocess.SortSplit(x,.05)
    xdiv4,x = Lathe.preprocess.SortSplit(x,.05)
    ydiv1,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv2,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv3,y = Lathe.preprocess.SortSplit(y,.05)
    ydiv4,y = Lathe.preprocess.SortSplit(y,.05)
    scalarlist18 = ydiv1 ./ xdiv1
    scalarlist19 = ydiv2 ./ xdiv2
    scalarlist20 = y ./ x
    # Now we sortsplit the x train
    xtdiv1,xt2 = Lathe.preprocess.SortSplit(xt,.05)
    xtdiv2,xt2 = Lathe.preprocess.SortSplit(xt2,.05)
    xtdiv3,xt2 = Lathe.preprocess.SortSplit(xt2,.05)
    xtdiv4,null = Lathe.preprocess.SortSplit(xt2,.05)
    range1 = minimum(xtdiv1):maximum(xtdiv1)
    range2 = minimum(xtdiv2):maximum(xtdiv2)
    range3 = minimum(xtdiv3):maximum(xtdiv3)
    range4 = minimum(xtdiv4):maximum(xtdiv4)
    range5 = minimum(null):maximum(null)
    returnlist = []
    for i in xt
        if i in range1
            res = i * rand(scalarlist1)
            append!(returnlist,res)
        elseif i in range2
            predlist = []
            res = i * rand(scalarlist2)
            append!(returnlist,res)

        elseif i in range3
            predlist = []
            res = i * rand(scalarlist3)
            append!(returnlist,res)
        elseif i in range4
            predlist = []
            res = i * rand(scalarlist4)
            append!(returnlist,res)
        else
            predlist = []
            res = i * rand(scalarlist20)
            append!(returnlist,res)
        end
    end
    return(returnlist)
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
# Model Type
mutable struct majBaseline
    y
end
#----  Callback
function pred_catbaseline(m,xt)
    y = m.y
    e = []
    mode = Lathe.stats.mode(xt)
    for i in xt
        append!(e,i)
    end

end
#
#----------------------------------------------
end
#================
Pipeline
    Module
================#
module pipelines

# Note to future self, or other programmer:
# It is not necessary to store these as constructors!
# They can just be strings, and use the model's X and Y!
using Lathe
mutable struct Pipeline
    model
    methods
    setting
end
function pipe_predict(pipe,xt)
    """ Takes a fit pipeline, and an X and predicts. """
    if pipe.setting == :CON
        model = pipe.model
        if typeof(pipe.methods) != Array
            [b = pipe.methods(b) for b in model.x]
            [b = pipe.methods(b) for b in xt]
        else
            for i in methods
                [b = i(b) for b in model.x]
                [b = i(b) for b in xt]
            end
        end
        y_pred = Lathe.models.predict(model,xt)
        return(y_pred)
    elseif pipe.setting == :CAT

    elseif pipe.setting == :MIX
    end
end
function save(pipe,filename)
    if typeof(pipe.model) == LinearRegression
        save(filename, Dict("m" => typeof(m),"x" => m.x,"y" => m.y))
    end
end
#----------------------------------------------
end
#==
This is the end of the main
module, nothing is to be written
beyond here
==#
end
