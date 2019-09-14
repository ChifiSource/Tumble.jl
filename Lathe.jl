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
#<----Median---->
function median(array)

end
#<----Mode---->
function mode(array)

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
function f_value(array

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
using Lathe.stats
sample = randsubseq(1:size(df,1), 0.05)
trainingset = df[sample, :]
notsample = [i for i in 1:size(df,1) if isempty(searchsorted(sample, i))]
testset = df[notsample, :]
function TrainTest(ratings::Array{Rating,1}, target_percentage=0.10)
  N = length(ratings)
  splitindex = round(Integer, target_percentage * N)
  shuffle!(ratings)
  return sub(ratings, splitindex+1:N), sub(ratings, 1:splitindex)
end
function StandardScalar(array)
    standardized = 0
    return(standardized)
end
function firstquar()

end
function secondquar()

end
function thirdquar()

end
function sampmed()

end
#-----------------------------
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
#----------------------------------------------
end
#==
This is the end of the main
module, nothing is to be written
beyond here
==#
end
