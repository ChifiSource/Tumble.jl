#<----Mean---->
"""
    ## Mean
    ### Description
      Returns the mean of an array.\n
      --------------------\n
    ### Input
      mean(x)\n
      --------------------\n
      #### Positional Arguments
      Array{Any} - x:: Array to obtain the mean of.
      --------------------\n
     ### Output
     mu:: The mean of the provided array.
       """
mean(x) = sum(x) / length(x)
#<----Median---->
"""
      Calculates the median (numerical center) of a given array.\n
      --------------------\n
      array = [5,10,15]\n
      median = Lathe.stats.median(array)\n
      println(median)\n
        10
       """
median(x::Array) = quantile(x, .5)

function quantile(x::Array, q::Real = .5)
    qdict = Dict(1 => .25, 2 => .5, 3 => .75)
    if q >= 1
        try
            q = qdict[q]
        catch
            throw(ArgumentError(" The quantile parameter is not set to a percentage, or quantile!"))
        end
    end
    sorted = sort(x)
    div = length(x)
    return(x[Int64(round(div * q))])
end
#<----Mode---->
"""
      Gives the digit most common in a given array\n
      --------------------\n
      array = [5,10,15,15,10,5,10]\n
      mode = Lathe.stats.mode(array)\n
      println(mode)\n
        10
       """
function mode(array)
    m = findmax(array)
    return(m)
end
#<----Variance---->
"""
      Gives the variance of an array..\n
      --------------------\n
      array = [5,10,15]\n
      variance = Lathe.stats.variance(array)\n
       """
function variance(array)
    me = mean(array)
    sq = sum(array) - me
    squared_mean = sq ^ 2
    return(squared_mean)
end
#<----Confidence Intervals---->
"""
      Returns the confidence intervals of given data.\n
      --------------------\n
      array = [5,10,15]\n
      confidence = .98\n
      low, high = Lathe.stats.confints(array,confidence)\n
      --------------------\n
      PARAMETERS\n
      confidence: Confidence is a float percentage representing the level of
      confidence required in your test. The confidence metric is used
      exclusively for calculating the interval.
       """
function confints(data, confidence=.95)
    mean = stats.mean(data)
    std = std(data)
    stderr = ste(data)
#    interval = stderr * scs.t.ppf((1 + confidence) / 2.0, n-1)
#    return (mean-interval, mean+interval)
end
#<----Standard Error---->
@doc """
      Calculates the Standard Error of an array.\n
      --------------------\n
      array = [5,10,15]\n
      ste = Lathe.stats.standarderror(array)"""
function ste(data)
    std = std(data)
    sample = length(data)
    ste = (std/sqrt(sample))
    return(ste)
end
#<----Standard Deviation---->
"""
      Calculates the Standard Deviation of a given array.\n
      --------------------\n
      array = [5,10,15]\n
      std = Lathe.stats.std(array)\n
       """
function std(array3)
    m = mean(array3)
    [i = (i-m) ^ 2 for i in array3]
    m = mean(array3)
    try
        m = sqrt(m)
    catch
        m = sqrt(Complex(m))
    end
    return(m)
end
"""
    ## Factorials
    ### Description
      Calculates the factorial of a number.\n
      --------------------\n
    ### Input
      fact(n)\n
      --------------------\n
      #### Positional Arguments
      Int64 - n:: The number for the factorial.\n
      --------------------\n
     ### Output
     f:: Factorial of n
       """
function fact(n)
    if n == 1
        return(1)
    else
        return n * fact(n-1)
    end
end

Σ(x) = sum(x)
mu(x) = mean(x)
