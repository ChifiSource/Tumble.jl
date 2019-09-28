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
================================#
module Lathe
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
#<----Median---->
function median(array)

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
function standardize(array)
    mean = sum(array)/length(array)
    sq = sum(array) - mean
    squared_mean = sq ^ 2
    standardized = sqrt(squared_mean)
    return(standardized)
end
#<----Confidence Intervals---->
function confiints(data, confidence=.95)
#    n = length(data)
#    mean = sum(data)/n
#    std = standardize(data)
#    stderr = standarderror(data)
#    interval = stderr * scs.t.ppf((1 + confidence) / 2.0, n-1)
#    return (mean-interval, mean+interval)
end
#<----Standard Error---->
function standarderror(data)
    std = standardize(data)
    sample = length(data)
    ste = (std/sqrt(sample))
    return(ste)
end
#<----Z score---->
function z(array)

end
function firstquar(array)
    m = median(array)
    q15 = array / m
    q1 = array / m
    return(q)
end
function secondquar(array)
    m = median(array)
    return(m)
end
function thirdquar(array)
    q = median(array)
    q = q * 1.5
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
    sampstd = standardize(data)
    grstd = standardize(grdata)
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
function student_t(sample,general)
    sampmean = mean(sample)
    genmean = mean(general)
    samples = length(sample)
    std = standardize(general)
    t = (sampmean - genmean) / (std / sqrt(samples))
    return(t)
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
#-------Model Metrics--------____________
function reg_sum(pred,gen)
    println("================")
    println("     Lathe.stats Regression Summary")
    println("     _______________________________")
end
function mae(actual,pred)
    l = length(actual)
    lp = length(pred)
    if l != lp
        throw(ArgumentError("The array shape does not match!"))
    end
    println(1,"X",l)
    m = mean(actual)
    prm = mean(pred)
    a = m-prm
    return(a)
end
function getPermutation(model)

end
#---------------------------
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
    n = Lathe.stats.nrow(data)
    idx = Random.shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
    return(test_idx,train_idx)
end
# SortSplit-------------
function SortSplit(data,at=.25,reverse=false)
    idx = sort(data)
    if reverse == true
        idx = sort(data,Sort.Reverse)
    end
    one = view(idx, 1:floor(Int, at*n))
    two = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
    return(one,two)
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
    for i in array
        min = minimum(array)
        max = maximum(array)
        x = i
        x = (x-min) / (max - min)
        append!(v,x)
    end
    return(v)
end
# ---- Arbitrary Rescalar ----
function ArbritatraryRescale(array)
    v = []
    for i in array
        a = minimum(array)
        b = maximum(array)
        x = i
        x = a + (x-a(x))*(b-a) / (b-a)
        append!(v,x)
    end
    return(v)
end
# ---- Mean Normalization ----
function MeanNormalization(array)
    avg = Lathe.stats.mean(array)
    first = True
    for i in array
        if first == True
            dtype = typeof(m)
            v = []
        end
        first = False
        x = i
        a = minimum(array)
        b = maximum(array)
        m = (x-avg) / (b-a)
        append!(v,x)
    end
end
# ---- Z Normalization ----
function z_normalize(array)
    q = Lathe.stats.standardize(array)
    avg = Lathe.stats.mean(array)
    v = []
    for i in array
        x = i
        y = (x-avg) / q
        append!(v,y)
    end
end
# ---- Unit L-Scale normalize ----
function Unit_LScale(array)
    print("Lathe 0.0.5")
    print("As of now, this feature is unavailable.")
end
#==========
Categorical
    Encoding
==========#
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
    println("________________")
    println("Current")
    println("    Usable")
    println("       Models")
    println("================")
#    println("--QuadRange--")
#    print("Ideal for use with continuous variables, uses")
#    print("4 ranges and mathematical gap to predict the")
#    print("outcome of Y. This results in a non-linear")
#    print("Prediction for data with high variance.")
#    print("---- Usage ----")
#    print("model = Lathe.models.QuadRange(x,y)")
#    print("ypr = Lathe.models.predict(model,Feature)")
    print("_________________________________")
#    println("--TurtleShell--")
#    println("--majBaseline--")
    println("--meanBaseline--")
    print("Basic model to get a baseline-accuracy to")
    print("improve upon")
    print("---- Usage ----")
    print("model = Lathe.models.meanBaseline(y)")
    print("ypr = predict(model,Feature)")
end
#Takes model, and X to predict, and returns a y prediction
function predict(m,x)
    if typeof(m) == QuadRange
        y_pred = pred_quadrange(m,x)
    end
    if typeof(m) == majBaseline
        y_pred = pred_catbaseline(m,x)
    end
    if typeof(m) == LinearRegression
        y_pred = pred_linearregression(m,x)
    end
    if typeof(m) == meanBaseline
        y_pred = pred_meanbaseline(m,x)
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
Quad
    Range
==#
# Model Type
mutable struct QuadRange
    x
    y
end
#----  Callback
function pred_quadrange(m,xt)
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
    # Empty lists for each range==

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
    len = length(xt)
    range1min = minimum(range1)
    range1max = maximum(range1)
    range2min = minimum(range2)
    range2max = maximum(range2)
    range3min = minimum(range3)
    range3max = maximum(range3)
    range4min = minimum(range4)
    range4max = maximum(range4)
    xrange1min = minimum(xrange1)
    xrange1max = maximum(xrange1)
    xrange2min = minimum(xrange2)
    xrange2max = maximum(xrange2)
    xrange3min = minimum(xrange3)
    xrange3max = maximum(xrange3)
    xrange4min = minimum(xrange4)
    xrange4max = maximum(xrange4)
    xrange1avg = Lathe.stats.mean(xrange1)
    xrange2avg = Lathe.stats.mean(xrange2)
    xrange3avg = Lathe.stats.mean(xrange3)
    xrange4avg = Lathe.stats.mean(xrange4)
    for i in xt
        pred = 2
        append!(e,pred)
    end
    return e
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
function pred_linearregression(m,xt)

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
Predictive
    Learning
        Models
================#
module pipelines
#
#----------------------------------------------
end
#==
This is the end of the main
module, nothing is to be written
beyond here
==#
end
