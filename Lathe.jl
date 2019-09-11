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
#<----Variance---->
function variance(array)
    mean = mean(array)
    sq = sum(array) - mean
    squared_mean = expo(sq,2)
    return(squared_mean)
end
#<----Standard Deviation---->
function standardize(array)
    mean = sum(array)/length(array)
    sq = sum(array) - mean
    squared_mean = expo(sq,2)
    standardized = sqrt(squared_mean)
    return(standardized)
end
#<----Exponential Expression---->
function expo(number,scalar)
    if scalar != 0
        newscale = scalar-1
        newnumber = number * number
        expo(newnumber,newscale)
    else
        return(number)
    end
end
#<----Confidence Intervals---->
function confiints(data, confidence=.95)
    n = length(data)
    mean = sum(data)/n
    std = standardize(data)
    stderr = standarderror(data)
    interval = stderr * scs.t.ppf((1 + confidence) / 2.0, n-1)
    return (mean-interval, mean+interval)
end
#<----Standard Error---->
function standarderror(data)
    std = standardize(data)
    sample = length(data)
    ste = (std/sqrt(sample))
    return(ste)
end
#-------Inferential-----------__________
#<----Inferential Summary---->
function inf_sum(data,grdata)
    #Doing our calculations
    t = student_t(data,grdata)
    f = f_test(samp,general)
    low,high = confiints(data)
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
    println(": ",gravg)
    println("N: ",len(grdata))
    println("x̅: ",avg)
    println("μ: ",gravg)
    println("s: ",sampstd)
    println("σ: ",grstd)
    println("var(X): ",var)
    println("σ2: ",grvar)
    println("Low Confidence interval: ",low)
    println("High Confidence interval: ",high)
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
    #F Statistic = variance of the group means / mean of the within group variances.
end
#<----F-Value---->
function f_value(array)

end
#-------Bayesian--------------___________
#<----Bayes Theorem---->
#P = prob, A = prior, B = Evidence,
function bay_ther(p,a,b)
    psterior = (p*(b|a) * p*(a)) / (p*b)
    return(psterior)
end
#-------Model Metrics--------____________
function reg_sum(pred,gen)
    println("================")
    println("     Lathe.stats Regression Summary")
    println("     _______________________________")
    println(": ",gravg)
    println("N: ",len(grdata))
    println("x̅: ",avg)
    println("μ: ",gravg)
    println("s: ",)
    println("Sample Variance: ",var)
    println("Group Variance: ",grvar)
    println("Low Confidence interval: ",low)
    println("High Confidence interval: ",high)
    println("Tp: ",t)
    println("Fp: ",f)
    println("================")
end
#---------------------------
end
#================
Preprocessing
     Module
================#
module preprocess

#----------------------------
end
#================
Predictive
    Learning
        Models
================#
module model
#==
Base
    functions
==#
function fit(model,x,y)
    model(x,y)
end
function predict(x)

end
#=====
Fluxxy
    Uses Flux
        For Advanced ML
=====#
module fluxxy
using Flux

end
#----------------------------------------------
#==
This is the end of the main
module, nothing is to be written
beyond here
==#
end
