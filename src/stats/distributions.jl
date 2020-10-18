"""
      Binomial Distribution is a distribution well known for its use in
           statistical tests and decision making models. In order to calculate
           binomial distribution, you will need the positives and size of your array.\n
      --------------------\n
      positives = 5\n
      n = 10\n
      r = binomial_dist(positives,n)
       """
function binomial_dist(positives, size; mode = :REC)
    # p = n! / x!(n-x!)*π^x*(1-π)^N-x
    n = size
    x = positives
    if mode != :REC
        factn = factorial(big(n))
        factx = factorial(big(x))
    else
        factn = fact(n)
        factx = fact(x)
    end
    return(factn / (factx * (n-x)))
end
# ---- Normal Distribution ----
"""
      Returns the normal distribution as an array.\n
      --------------------\n
      array = [5,10,15]\n
      dist = normal_dist(array)
       """
function NormalDist(array)
    σ = std(array)
    μ = mean(array)
    apply(xt) = [i = (i-μ) / σ for i in xt]
    cdf = ""
    (var) ->(σ;μ;cdf)
end
# ---- T distribution ----
"""
      Returns the T distribution as an array.\n
      --------------------\n
      array = [5,10,15]\n
      dist = t_dist(sample, general)
       """
function TDist(general)
    norm = NormalDist(general)
    general = norm.apply(general)
    μ = mean(general)
    N = length(general)
    apply(xt) = (mean(norm.apply(xt)) - μ) / (std(norm.apply(xt)) / sqrt(N))
    cdf = ""
    (distribution)->(μ;N;apply;cdf)
end
