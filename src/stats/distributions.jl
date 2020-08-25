"""
      Binomial Distribution is a distribution well known for its use in
           statistical tests and decision making models. In order to calculate
           binomial distribution, you will need the positives and size of your array.\n
      --------------------\n
      positives = 5\n
      n = 10\n
      r = binomial_dist(positives,n)
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
      Returns the normal distribution as an array.\n
      --------------------\n
      array = [5,10,15]\n
      dist = normal_dist(array)
       """
function normal_dist(array)
    q = Lathe.stats.std(array)
    avg = Lathe.stats.mean(array)
    v = [i = (i-avg) / q for i in array]
    return(v)
end
# ---- T distribution ----
"""
      Returns the T distribution as an array.\n
      --------------------\n
      array = [5,10,15]\n
      dist = t_dist(array)
       """
function t_dist(sample, general)
    x̅ = mean(sample)
    μ = mean(general)
    s = std(sample)
    N = length(general)
    arr = [obso = (x̅ - μ) / (s / sqrt(N)) for obso in sample]
    return(arr)
end
