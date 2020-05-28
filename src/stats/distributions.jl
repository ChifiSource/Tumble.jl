# <---- Chi Distribution --->
"""
      FUNCTION NOT YET WRITTEN\n
      Chi Distribution in another well-known distribution well known for being
      used in statistical tests.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """
function chi_dist(x,e)
    #
end
"""Function Not yet written"""
function bernoulli_dist()
    # P(x) = P^x(1-P)^1-x for x=0 eller 1
end
"""
      Binomial Distribution is a distribution well known for its use in
           statistical tests and decision making models.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """
function binomial_dist(positives,size)
    # p = n! / x!(n-x!)*π^x*(1-π)^N-x
    n = size
    x = positives
    factn = factorial(big(n))
    factx = factorial(big(x))
    nx = factn / (factx * (n-x))
    return(nx)
end
# ---- Normal Distribution ----
"""
      Standard Scalar z-score normalizes a feature.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.StandardScalar(array)\n
       """
function normal_dist(array)
    q = Lathe.stats.std(array)
    avg = Lathe.stats.mean(array)
    v = [i = (i-avg) / q for i in array]
    return(v)
end
# ---- T distribution ----
function t_dist(sample, general)
    x̅ = mean(sample)
    μ = mean(general)
    s = std(sample)
    N = length(general)
    arr = [obso = (x̅ - μ) / (s / sqrt(N)) for obso in sample]
    return(arr)
end
