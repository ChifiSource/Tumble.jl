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
#<----T Value---->
function t_test(sample,general)
    sampmean = mean(sample)
    genmean = mean(general)
    samples = length(sample)
    std = standardize(general)
    t = (sampmean - genmean) / (std / sqrt(samples))
    return(t)
end

#<---- F-Test---->
function f_test(array)

end
#<----F-Value---->
function f_stat(array)

end
#-------Bayesian--------------___________
#<----Bayes Theorem---->
#P = prob, A = prior, B = Evidence,
function bay_ther(p,a,b)
    psterior = (p*(b|a) * p*(a)) / (p*b)
    return(psterior)
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
module predmodel
#==
Regression
    Models
==#
#-----Regression-----------------_____________
#<----Univariate Regression---->
#==WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP
function UnivariateRegression{T<:AbstractFloat}(loss::UnivariateLoss,
                                                X::StridedMatrix{T},
                                                Y::StridedVector;
                                                bias::Real=0.0)
    d, n = size(X)
    length(Y) == n || throw(DimensionMismatch())
    UnivariateRegression{typeof(loss), T, typeof(X), typeof(Y)}(
        loss, d, n, convert(T, bias), X, Y)
end
#<----Linear Regression---->
LinearRegression{T<:AbstractFloat}(X::StridedMatrix{T}, y::StridedVector{T}; bias::Real=0.0) =
    UnivariateRegression(SqrLoss(), X, y; bias=bias)
end
#<----Logistic Regression---->
LogisticRegression{T<:AbstractFloat}(X::StridedMatrix{T}, y::StridedVector; bias::Real=0.0) =
    UnivariateRegression(LogisticLoss(), X, convert(Vector{T}, y); bias=bias)
end
WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP==#

#----------------------------------------------
end
#==
This is the end of the main
module, nothing is to be written
beyond here
==#
end
