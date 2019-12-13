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
================================#

module Lathe

# <------- PARTS ----->
include("nlp.jl")
include("pipelines.jl")
# <------- PARTS ----->
# <------- DEPS ----->
using DataFrames
using Random
# <------- DEPS ----->
#================
Stats
    Module
================#

module stats
#<----Mean---->
@doc """
      Calculates the mean of a given array.\n
      array = [5,10,15]\n
      mean = Lathe.stats.mean(array)\n
      println(array)\n
        10
       """ ->
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
    x̄ = mean(array)
    σ = std(array)
    return map(x -> (x - x̄) / σ, array)
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
    sortedar = sort(array,rev=rev)
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
    d = var1 .- var2
    d̄ = mean(x)
end
#<---- Binomial Distribution ---->
function binomial_prob(positives,size)
    # p = n! / x!(n-x!)*π^x*(1-π)^N-x
    n = size
    x = positives
    factn = factorial(n)
    factx = factorial(x)
    nx = factn / (factx * (n-x))
    return(nx)
end
#<---- Correlations ---->
# - Spearman
function spearman(var1,var2)
    rgX = getranks(var1)
    rgY = getranks(var2)
    ρ = rgX*rgY / (std(rgX)*std(rgY))
    return(ρ)
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
# it is tough to calculate -> is it really needed?
    # A little less necessary, as its certainly not the most useful,
    # But this stats library could serve as a foundation for models that
    # Utilize Chi-Distributions, and although I wouldn't say having
    # A function to do so is urgent, It definitely would be cool,
    # Rather than an end user having to add another package just
    # To do one or two things, if you know what I mean.
    # But certainly there are other more important things to get through
    # Before 1.0, and I wouldn't consider any of these statistics incredibly
    # Necessary, but the template is there for what I want to include,
    # So people adding the module now can kindof know what to expect.
    # So hopefully that answers your question!
end
#<---- Chi-Square ---->
function chisq(var1,var2)
    chistat(obs, exp) = (obs - exp)^2/exp
    return chistat.(x, e) |> sum
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
    ans = binomialprob(positives,totallen)
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
#=========================
Distributions section!!!!!
~Added Lathe 0.0.6 ~
=========================#
function bernoulli_dist()
    # P(x) = P^x(1-P)^1-x for x=0 eller 1
end
function binomial_dist()
    # P(X) = nCxp^x(1-p)^n-x
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
function OneHotEncode(array::Number)
    flatarr = Iterators.flatten(array)
    len = size(flatarr, 2)
    poslen = size(unique(flatarr), 2)
    out = Array{Number}(undef, len, poslen)
    for i in 1:len
        el = flatarr[i]
        idx = findall(x -> x == el, flatarr |> unique)[1][2]
        out[i, idx] = 1
    end
    return(out)
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
    if typeof(m) == MultipleLinearRegression
        y_pred = pred_multiplelinearregression(m,x)
    end
    if typeof(m) == Pipeline
        for step in m.steps
            x = step(x)
        end
        ypr = Lathe.models.predict(m.model,x)

        return(ypr)
    end
    return(y_pred)
end
#===========
Accessories
===========#
# The help function:
function help(args)
    if typeof(args) == LogisticRegression
        println("Logistic")
    elseif typeof(args) == meanBaseline
        println("MeanBaseline")
    end
end
mutable struct Pipeline
    steps
    model
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
function pred_regressiontree(m,xt)
    # x = q1(r(floor:q1)) |x2 = q2(r(q1:μ)) |x3 = q3(r(q2:q3)) |x4 q4(r(q3:cieling))
    # y' = q1(x * (a / x)) | μ(x * (a / x2)) | q3(x * (a / x3) | q4(x * (a / x4))
    # Original 4 quartile math ^^
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
    for z in xt
        r = 0
        predavg = []
        for i in z
            m = LinearRegression(z,m.y)
            pred = predict(m,z)
            append!(predavg,pred)
        end
        append!(y_pred,predavg)
    end
    len = length(y_pred[1])
    yprl = length(y_pred)
    pr = []
    numbers = collect(1:yprl)
    oddsonly = numbers[numbers .% 2 .== 0]
    oddsonly = filter!(e->e≠0,oddsonly)
    if yprl in oddsonly
        truonly = true
    else
        truonly = false
    end
        for z in oddsonly
            cp = z + 1
            for i in 1:len
                s = z[i]
                v = cp[i]
                d = Lathe.stats.mean([s,v])
                append!(pr,d)
            end
        end
        if truonly == true
            fn = maximum(oddsonly)
            z = fn
            cp = fn + 1
            for i in 1:len
                s = z[i]
                v = cp[i]
                d = Lathe.stats.mean([s,v])
                append!(pr,d)
            end
        end
    return(pr)
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
    return(xt)
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
    # (LLSQ Base, may allow changing with hyper-parameters
    # in the future)
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
    #    (For Loop)
    xmean = Lathe.stats.mean(xt)
    y_pred = []
    for i in xt
        pred = (slope*i) + b + (i - xmean)
        append!(y_pred,pred)
    end
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
#==
This is the end of the main
module, nothing is to be written
beyond here
==#
end
