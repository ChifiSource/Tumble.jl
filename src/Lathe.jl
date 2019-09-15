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
using DataFrames
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
    if typeof(data) == DataFrame
        nrow(data)
    else
        x = 0
        for i in data
            x = x+1
        end
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
function sampmed(array)

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
#<---- T Value---->
function t_value(array)

end
#<---- F-Test---->
function f_test(sample,general)
    totvariance = variance(general)
    sampvar = variance(sample)
    f =  sampvar / totvariance
end
#<----F-Value---->
#function f_value(array

#end
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
#---------------------------
end
#================
Preprocessing
     Module
================#
module preprocess
using Random
using Lathe

function TrainTest(data, at = 0.7)
    n = Lathe.stats.nrow(data)
    idx = Random.shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
    return(test_idx,train_idx)
end
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
function Rescalar(array)
    v = AbstractArray
    for i in array
        min = floor(array)
        max = ceiling(array)
        x = i
        x = (x-min) / (max - min)
        append!(x,v)
    end
    return(v)
end
function ArbritatraryRescale(array)
    v = AbstractArray
    for i in array
        a = floor(array)
        b = ceiling(array)
        x = i
        x = a + (x-a(x))*(b-a) / (b-a)
        append!(x,v)
    end
    return(v)
end
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
        a = floor(array)
        b = ceiling(array)
        m = (x-avg) / (b-a)
        append!(x,v)
    end
end
function z_normalize(array)
    q = Lathe.stats.standardize(array)
    avg = Lathe.stats.mean(array)
    v = AbstractArray
    for i in array
        x = i
        y = (x-avg) / q
    end
end
function Unit_LScale(array)

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
#Show models shows all the models that are stable
#And ready for use in the library
function showmodels()
    println("________________")
    println("Current")
    println("    Usable")
    println("       Models")
    println("================")
    println("turtleshell")
    println("baseline")
end
function predict(m,x)
    if typeof(m) == TurtleShell
        pred_turtleshell(m,x)
    end
    if typeof(m) == Baseline
        pred_baseline(m,x)
    end
    if typeof(m) == LinearRegression
        pred_linearregression(m,x)
    end
end
#==
Turtle
    Shell
==#
# Model Type
mutable struct TurtleShell
    x
    y
end
# Prediction Function
function pred_turtleshell(m,xt)
    x = m.x
    y = m.y
end
#==
Baseline
==#
# Model Type
mutable struct Baseline
    y
end
# Prediction Function
function pred_baseline(m,xt)
    y = m.y
    r = length(xt)
    e = []
    mode = Lathe.stats.mode(xt)
    for i in xt
        append!(i,e)
        while r != 0
            r = r -1
        end
    end

end
#==
Linear
    Regression
==#
mutable struct LinearRegression
    x
    y
end
function pred_linearregression(m,xt)

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
