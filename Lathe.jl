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
#<----T Value---->
function t_test(sample,general)
    sampmean = mean(sample)
    genmean = mean(general)
    samples = length(sample)
    std = standardize(general)
    t = (sampmean - genmean) / (std / sqrt(samples))
    return(t)
end
#<----Bayes Theorem---->
function bay_ther(p,a,b)
    psterior = (p*(b|a) * p*(a)) / (p*b)
    return(psterior)
end
#<----Confidence Intervals---->
function confiint(data, confidence=.95)
    n = length(data)
    mean = sum(data)/n
    #stderr = scs.sem(data)
    #interval = stderr * scs.t.ppf((1 + confidence) / 2.0, n-1)
    #return (mean , mean-interval, mean+interval)
end
#================
Preprocessing
     Module
================#
module preprocess

end
#================
Machine
    Learning
        Models
================#
module model

end
#==
This is the end of the main
module, nothing is to be written
beyond here
==#
end
