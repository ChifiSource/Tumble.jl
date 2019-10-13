#===============================
<-----------Lathe.jl----------->
Programmed by Emmett Boudreau
    <emmett@emmettboudreau.com>
        <http://emmettboudreau.com>
GNU General Open Source License
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
#<----Nrow counts number of iterations---->
function nrow(data)
        x = 0
        for i in data
            x = x+1
        end
        return(x)
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
#<----Standard Deviation---->
#function std(ar)
#    ms = sum(ar)/length(ar)
#    l = []
#    for i in ar
#        subtr = (i - ms) ^ 2
#        append!(l,subtr)
#    end
#    me = sum(l)/length(l)
#    squared_mean = me ^ 2
#    standardized = sqrt(squared_mean)
#    return(standardized)
#end
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
function std(array)
    avg = mean(array)
    l = []
    for i in array
        subtr = (i-avg) ^ 2
        append!(l,subtr)
    end
    me = mean(l)
    me = me ^ 2
    standard = sqrt(me)
    return(standard)
end
#<---- Correlation Coefficient --->
function correlationcoeff(x,y)
    n = length(x)
    yl = length(y)
    if n != yl
        throw(ArgumentError("The array shape does not match!"))
    end
    xy = x .* y
    x2 = x .^ 2
    y2 = y .^ 2
    sx = sum(x)
    sy = sum(y)
    sxy = sum(xy)
    sx2 = sum(x2)
    sy2 = sum(y2)
    numer = (n * sxy) - (sx * sy)
    denom = sqrt(n*sx2 - sx^2) * (n*sy2 - sy^2)
    corrcoff = numer / denom
    return(corrcoff)
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
    t = student_t(data,grdata)
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
    println("var(X): ",var)
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
    std = std(general)
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
function pearson(var1,var2)

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
function binomialdist(positives,negatives,zeros)
    # p = n! / x!(n-x!)*π^x*(1-π)^N-x
    n = positives + negatives + zeros
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
    totalnegs = length(negatives)
    totalpos = length(positives)
    totalzer = length(zeros)
    totallen = length(sets)
    ans = binomialdist(positives,negatives,zeros)
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
        maeunf = maeunf - (maeunf - maeunf)
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
function TrainTest(data, at = 0.7)
    n = length(data)
    idx = Random.shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
    return(test_idx,train_idx)
end
# DataFrames TestTrainSplit -----
function DfTrainTest(data, at = 0.7)
    n = nrow(data)
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
# Test Train Val Split----
function TrainTestVal(data, at = 0.6,valat = 0.2)
    n = Lathe.stats.nrow(data)
    idx = Random.shuffle(1:n)
    train = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train,:], data[test_idx,:]
    # ~Repeats to split test data~
    n = Lathe.stats.nrow(test)
    idx = Random.shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    val_idx = view(idx, (floor(Int, at*n)+1):n)
    train[train_idx,:], train[val_idx,:]
    return(test_idx,train_idx,val_idx)
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
    return(y_pred)
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
        range4 = y4
        # Split the x train
        x1,xrange1 = Lathe.preprocess.SortSplit(x)
        x2,xrange2 = Lathe.preprocess.SortSplit(x1)
        x3,xrange3 = Lathe.preprocess.SortSplit(x2)
        xrange4 = x3
        # Fitting the 4 linear regression models ---->
        regone = LinearRegression(xrange1,range1)
        regtwo = LinearRegression(xrange2,range2)
        regthree = LinearRegression(xrange3,range3)
        regfour = LinearRegression(xrange4,range4)
        # Split the train Data
        xt1,xtrange1 = Lathe.preprocess.SortSplit(xt)
        xt2,xtrange2 = Lathe.preprocess.SortSplit(xt1)
        xt3,xtrange3 = Lathe.preprocess.SortSplit(xt2)
        xtrange4 = xt3
        # Get min-max
        xtrange1min = minimum(xtrange1)
        xtrange1max = maximum(xtrange1)
        xtrange2min = minimum(xtrange2)
        xtrange2max = maximum(xtrange2)
        xtrange3min = minimum(xtrange3)
        xtrange3max = maximum(xtrange3)
        xtrange4min = minimum(xtrange4)
        xtrange4max = maximum(xtrange4)
        # Ranges for ifs
        condrange1 = (xtrange1min:xtrange1max)
        condrange2 = (xtrange2min:xtrange2max)
        condrange3 = (xtrange3min:xtrange3max)
        condrange4 = (xtrange4min:xtrange4max)
        # This for loop is where the dimension's are actually used:
        for i in xt
            if i in condrange1
                ypred = predict(regone,i)
            end
            if i in condrange2
                ypred = predict(regtwo,i)
            end
            if i in condrange3
                ypred = predict(regthree,i)
            end
            if i in condrange4
                ypred = predict(regfour,i)
            end
            append!(e,ypred)
        end
        return(e)
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
    # Empty array:
    ypred = []
    for i in xt
        yp = a+(b*i)
        append!(ypred,yp)
    end
    return(ypred)
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
Linear
    Scalar
==#
mutable struct ExponentialScalar
    x
    y
    n_predictors
end
function pred_exponentialscalar(m,xt)
    x = m.x
    y = m.y
    xdiv1,x = Lathe.preprocess.SortSplit(x)
    xdiv2,x = Lathe.preprocess.SortSplit(x)
    xdiv3,x = Lathe.preprocess.SortSplit(x)
    xdiv4,x = Lathe.preprocess.SortSplit(x)
    ydiv1,y = Lathe.preprocess.SortSplit(y)
    ydiv2,y = Lathe.preprocess.SortSplit(y)
    ydiv3,y = Lathe.preprocess.SortSplit(y)
    ydiv4,y = Lathe.preprocess.SortSplit(y)
    scalarlist1 = ydiv1 ./ xdiv1
    scalarlist2 = ydiv2 ./ xdiv2
    scalarlist3 = ydiv3 ./ xdiv3
    scalarlist4 = ydiv3 ./ xdiv3
    # Now we sortsplit the x train
    xtdiv1,xt2 = Lathe.preprocess.SortSplit(xt)
    xtdiv2,xt2 = Lathe.preprocess.SortSplit(xt2)
    xtdiv3,xt2 = Lathe.preprocess.SortSplit(xt2)
    xtdiv4,null = Lathe.preprocess.SortSplit(xt2)
    range1 = minimum(xtdiv1):maximum(xtdiv1)
    range2 = minimum(xtdiv2):maximum(xtdiv2)
    range3 = minimum(xtdiv3):maximum(xtdiv3)
    range4 = minimum(xtdiv4):maximum(xtdiv4)
    returnlist = []
    for i in xt
        if i in range1
            res = i * rand(scalarlist1)
            append!(returnlist,res)
        end
        if i in range2
            res = i * rand(scalarlist2)
            append!(returnlist,res)
        end
        if i in range3
            res = i * rand(scalarlist3)
            append!(returnlist,res)
        end
        if i in range4
            res = i * rand(scalarlist4)
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
module Pipelines
#=================================================
#
# Note to future self, or other programmer:
# It is not necessary to store these as constructors!
# They can just be strings, and use the model's X and Y!
using Lathe
mutable struct Pipeline
    model
    categoricalenc
    contenc
    imputer
end
mutable struct fitpipeline
    pipeline
    x
    y
end
function pipelinebuilder()
    println("== Lathe.JL Pipeline Builder ==")
    println("- Select a model -")
    m = readline()
    if m == "LinearRegression"
        model = Lathe.models.LinearRegression
    else
        println(m," is not a valid model.")
        println("Pipeline is continuing without a model.")
        model = false
    end
    println("- Select a categorical encoder -")
    cat = readline()
    println(m," is your selected categorical encoder")
    println("- Select a continous encoder -")
    m = readline()
    println(m," is your selected Continous Encoder")
    println("- Select an imputer -")
    imputer = false
    pipl = Pipeline(model,cat,m,imputer)
    println("Your new pipeline is officially created!")
    return(pipl)
end
function fitpipeline(Pipeline,x,y)
    pipl = fitpipeline(Pipeline,x,y)
end
function predict(fitpipeline,xt)
    if typeof(fitpipeline) == Pipeline
        throw(ArgumentError("This pipeline is not yet fitted!"))
    end
    # Preprocessing
    if fitpipeline.Pipeline.contenc == "Rescalar"
        fitpipeline.x = Lathe.preprocess.Rescalar(fitpipeline.x)
    end
    if typeof(fitpipeline.Pipeline.model) == Lathe.models.LinearRegression
        model = Lathe.models.LinearRegression(fitpipeline.x,fitpipeline.y)
        ypr = Lathe.models.predict(model,xt)
    end
end
=============================================#
#----------------------------------------------
end
#==
This is the end of the main
module, nothing is to be written
beyond here
==#
end
