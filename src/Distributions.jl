
# <---- Chi Distribution --->
@doc """
      FUNCTION NOT YET WRITTEN\n
      Chi Distribution in another well-known distribution well known for being
      used in statistical tests.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """
function chidist(x,e)
    #
end
function bernoulli_dist()
    # P(x) = P^x(1-P)^1-x for x=0 eller 1
end
@doc """
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
# ---- Z Normalization ----
@doc """
      Standard Scalar z-score normalizes a feature.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.StandardScalar(array)\n
       """
function normal(array)
    q = Lathe.stats.std(array)
    avg = Lathe.stats.mean(array)
    v = [i = (i-avg) / q for i in array]
    return(v)
end
export normal
export chidist
export binomial_dist
export bernoulli_dist
