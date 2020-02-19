#================
Stats
    Module
================#
@doc """
      |====== Lathe.stats ======\n
      | ~~~~~~~~~~ Base ~~~~~~~~~~~\n
      |_____stats.mean(array)\n
      |_____stats.mode(array)\n
      |_____stats.variance(array)\n
      |_____stats.confiints(data,confidence = .95)\n
      |_____stats.standarderror(array)\n
      |_____stats.std(data)\n
      |_____stats.correlationcoeff(x,y)\n
      |_____stats.z(array)\n
      |_____stats.firstquar(array)\n
      |_____stats.secondquar(array)\n
      |_____stats.thirdquar(array)\n
      |_____stats.getranks(array,rev = false)\n
      | ~~~~~~~~~~ Inferential ~~~~~~~~~~~\n
      |_____stats.independent_t(sample,general)\n
      |_____stats.paired_t(array)\n
      |_____stats.spearman(var1,var2)\n
      |_____stats.pearson(x,y)\n
      |_____stats.chisqu(array)\n
      |_____stats.sign(array)\n
      |_____stats.f_test(sample,general)\n
      |_____stats.anova(arra)\n
      | ~~~~~~~~~~ Bayesian ~~~~~~~~~~~\n
      |_____stats.bay_ther(p,a,b)\n
      |_____stats.cond_prob(p,a,b)\n
      | ~~~~~~~~~~ Distributions ~~~~~~~~~~~\n
      |_____stats.bournelli_dist(array)\n
      |_____stats.binomial_dist(positives,size)\n
       """ ->
module stats
#<----Mean---->
@doc """
      Calculates the mean of a given array.\n
      --------------------\n
      array = [5,10,15]\n
      mean = Lathe.stats.mean(array)\n
      println(mean)\n
        10
       """ ->
function mean(array)
    observations = length(array)
    average = sum(array)/observations
    return(average)
end
#<----Median---->
@doc """
      Calculates the median (numerical center) of a given array.\n
      --------------------\n
      array = [5,10,15]\n
      median = Lathe.stats.median(array)\n
      println(median)\n
        10
       """
function median(array)
    n = length(array)
    half = n / 2
    current = 1
    sorted = sort!(array,rev = false)
    median = 0
    for i in sorted
        if current >= half
            median = i
        else
        current = current + 1
    end
    end
    return(median)
end
#<----Mode---->
@doc """
      Gives the digit most common in a given array\n
      --------------------\n
      array = [5,10,15,15,10,5,10]\n
      mode = Lathe.stats.mode(array)\n
      println(mode)\n
        10
       """ ->
function mode(array)
    m = findmax(array)
    return(m)
end
#<----Variance---->
@doc """
      Gives the variance of an array..\n
      --------------------\n
      array = [5,10,15]\n
      variance = Lathe.stats.variance(array)\n
       """ ->
function variance(array)
    me = mean(array)
    sq = sum(array) - me
    squared_mean = sq ^ 2
    return(squared_mean)
end
#<----Confidence Intervals---->
@doc """
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
       """ ->
function confiints(data, confidence=.95)
    mean = mean(data)
    std = std(data)
    stderr = standarderror(data)
#    interval = stderr * scs.t.ppf((1 + confidence) / 2.0, n-1)
#    return (mean-interval, mean+interval)
end
#<----Standard Error---->
@doc """
      Calculates the Standard Error of an array.\n
      --------------------\n
      array = [5,10,15]\n
      ste = Lathe.stats.standarderror(array)\n
       """ ->
function standarderror(data)
    std = std(data)
    sample = length(data)
    ste = (std/sqrt(sample))
    return(ste)
end
#<----Standard Deviation---->
@doc """
      Calculates the Standard Deviation of a given array.\n
      --------------------\n
      array = [5,10,15]\n
      std = Lathe.stats.std(array)\n
       """ ->
function std(array3)
    m = mean(array3)
    [i = (i-m) ^ 2 for i in array3]
    m = mean(array3)
    m = sqrt(Complex(m))
    return(m)
end
#<---- correlation Coefficient --->
@doc """
      Calculates the Correlation Coeffiecient of between two features\n
      --------------------\n
      x = [5,10,15]\n
      y = [5,10,15]\n
      r = Lathe.stats.correlationcoeff(x,y)\n
       """ ->
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
@doc """
      Calculates the Z score of a given array.\n
      --------------------\n
      array = [5,10,15]\n
      z = Lathe.stats.z(array)\n
       """ ->
function z(array)
    x̄ = mean(array)
    σ = std(array)
    return map(x -> (x - x̄) / σ, array)
end
#<----Quartiles---->
# - First
@doc """
      Returns the point in an array located at 25 percent of the sorted data.\n
      --------------------\n
      array = [5,10,15]\n
      q1 = Lathe.stats.firstquar(array)\n
       """ ->
function firstquar(array)
    m = median(array)
    q1 = array * .5
    return(q1)
end
# - Second(median)
@doc """
      Returns the point in an array located at 50 percent of the sorted data.
      The second quartile is also known as the median, or the middle of the sorted data.\n
      --------------------\n
      array = [5,10,15]\n
      q2 = Lathe.stats.secondquar(array)\n
       """ ->
function secondquar(array)
    m = median(array)
    return(m)
end
# - Third
@doc """
      Returns the point in an array located at 75 percent of the sorted data.\n
      --------------------\n
      array = [5,10,15]\n
      q2 = Lathe.stats.secondquar(array)\n
       """ ->
function thirdquar(array)
    q = median(array)
    q3 = q * 1.5
    return(q3)
end
# <---- Rank ---->
@doc """
      Ranks indices in an array based on quantitative weight (count of the
      numbers) and returns a new array of the ranks of each column. This
      function is made primarily for the Wilcox Rank-Sum test.\n
      --------------------\n
      array = [5,10,15]\n
      q2 = Lathe.stats.secondquar(array)\n
       """ ->
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
#<----T Test---->
# - Independent
@doc """
      Performs an independent T test.\n
      --------------------\n
      sample = [5,10,15]
      general = [15,25,35]\n
      t = Lathe.stats.independent_t(sample,general)\n
       """ ->
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
@doc """
    THIS FUNCTION IS NOT YET WRITTEN\n
      Paired T (Dependent T) is a T-test that doesn't require a sample.\n
      --------------------\n
      array = [5,10,15]\n
      q2 = Lathe.stats.paired_t(var1,var2)\n
       """ ->
function paired_t(var1,var2)
    d = var1 .- var2
    d̄ = mean(x)
end
#<---- Correlations ---->
# - Spearman
@doc """
      Returns a probability using a Spearman correlation.\n
      --------------------\n
      var1 = [5,10,15]\n
      var2 = [5,4,3,5,6]\n
      r = Lathe.stats.spearman(var1,var2)\n
       """ ->
function spearman(var1,var2)
    rgX = getranks(var1)
    rgY = getranks(var2)
    ρ = rgX*rgY / (std(rgX)*std(rgY))
    return(ρ)
end
# - Pearson
@doc """
      Returns a probability using a Pearson correlation.\n
      --------------------\n
      x = [5,10,15]\n
      y = [5,4,3,5,6]\n
      r = Lathe.stats.spearman(x,y)\n
       """ ->
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
#<---- Chi-Square ---->
@doc """
      Returns a probability using a chi squared distribution.\n
      --------------------\n
      var1 = [5,10,15]\n
      var2 = [5,4,3,5,6]\n
      p = Lathe.stats.chisq(var1,var2)\n
       """ ->
function chisq(var1,var2)
    chistat(obs, exp) = (obs - exp)^2/exp
    return chistat.(x, e) |> sum
end
#<---- ANOVA ---->
@doc """
      FUNCTION NOT YET WRITTEN\n
      Anova is used to analyze variance in an array\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """ ->
function anova(var1,var2)

end
#<---- Wilcoxon ---->
# - Wilcoxon Rank-Sum Test
function wilcoxrs(var1,var2)
    #Hash
end
@doc """
      FUNCTION NOT YET WRITTEN\n
      Wilcox Sum Rank Tests are used to determine a probability with ranks\n
      --------------------\n
      var1 = [5,10,15]\n
      var2 = [19,25,30]\n
      p = Lathe.stats.wilcoxsr(var1,var2)\n
       """ ->
function wilcoxsr(var1,var2)
    #Hash
end
#<---- Sign Test ---->
@doc """
      The Sign test determines correlation through negative and positive
      placement with binomial distribution.\n
      --------------------\n
      var1 = [5,10,15]\n
      var2 = [19,25,30]\n
      p = Lathe.stats.sign(var1,var2)\n
       """ ->
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
    ans = binomial_dist(positives,totallen)
    return(ans)
end
#<---- F-Test---->
@doc """
      An F test returns a probability of correlation, and is used similarly
      to a T test.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """ ->
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
@doc """
      Performs Bayesian Conditional Probability and returns probability.\n
      --------------------\n
      prob = .50\n
      prior = .20\n
      evidence = .30\n
      p = Lathe.stats.cond_prob(prob,prior,evidence)\n
       """ ->
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
@doc """
      Binomial Distribution is a distribution well known for its use in
           statistical tests and decision making models.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """ ->
function binomial_dist(positives,size)
    # p = n! / x!(n-x!)*π^x*(1-π)^N-x
    n = size
    x = positives
    factn = factorial(big(n))
    factx = factorial(big(x))
    nx = factn / (factx * (n-x))
    return(nx)
end
# <---- Chi Distribution --->
@doc """
      FUNCTION NOT YET WRITTEN\n
      Chi Distribution in another well-known distribution well known for being
      used in statistical tests.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """ ->
function chidist(x,e)
    #
end
#================
Model
    Validation
        Module
================#
@doc """
      |====== Lathe.validate ======\n
      |____________/ Metrics ___________\n
      |_____validate.mae(actual,pred)\n
      |_____validate.r2(actual,pred)\n
      |___________/ Feature-Selection ___________\n
      |_____validate.permutation(model)
       """
#-------Model Metrics--------____________
using Lathe
## <---- Mean Absolute Error ---->
@doc """
      Mean absolute error (MAE) subtracts two arrays and averages the
      difference.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """
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
# <---- Mean Squared Error ---->
@doc """
      Mean Square error (MSE) subtracts two arrays, squares the
      difference, and averages the result\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """
function mse(y,ŷ)
    diff = y .- ŷ
    diff = diff .^ 2
    Σdiff = sum(diff)
    return(Σdiff)
end
# <---- R Squared ---->
@doc """
      R squared is the correlation coefficient of regression, and is found
      by squaring the correlation coefficient.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """
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
@doc """
      FUNCTION NOT YET WRITTEN\n
      Permutations are used when feature importance is taken into account for a
          model.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """
function permutation(model)

end
#---------------------------
end
