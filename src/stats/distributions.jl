using Distributions: TDist, Uniform
import Distributions: cdf
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
    ## Normal Distribution
    ### Description
      Calculates the normal distribution of an array.\n
      --------------------\n
    ### Input
      NormalDist(x)\n
      --------------------\n
      #### Positional Arguments
      Array{Any} - x:: Array for which the normal distribution should use the
      data from.\n
      --------------------\n
     ### Output
     norm:: A Lathe distribution\n
     ---------------------\n
     ### Functions
     Distribution.apply(xt) :: Applies the distribution to xt\n
     Distribution.cdf(statistic, alpha, dof) :: Applies the distribution's
     corresponding cummulitive distribution function.\n
     ---------------------\n
     ### Data
     σ :: Standard Deviation of the input data.\n
     μ :: Mean of the input data.
       """
function NormalDist(array)
    σ = std(array)
    μ = mean(array)
    apply(xt) = [i = (i-μ) / σ for i in xt]v
    (var) ->(σ;μ;cdf;apply)
end
# ---- T distribution ----
"""
    ## T Distribution
    ### Description
      Calculates the T distribution of an array.\n
      --------------------\n
    ### Input
      TDist(x)\n
      --------------------\n
      #### Positional Arguments
      Array{Any} - x:: Array for which the T distribution should use the
      data from.\n
      --------------------\n
     ### Output
     t:: A Lathe distribution\n
     ---------------------\n
     ### Functions
     Distribution.apply(xt) :: Applies the distribution to xt\n
     Distribution.cdf(statistic, alpha, dof) :: Applies the distribution's
     corresponding cummulitive distribution function.\n
     ---------------------\n
     ### Data
     μ :: Mean of the input data.\n
     N :: The length of the input data.
       """
function T_Dist(general)
    norm = NormalDist(general)
    general = norm.apply(general)
    μ = mean(general)
    N = length(general)
    apply(xt) = (mean(norm.apply(xt)) - μ) / (std(norm.apply(xt)) / sqrt(N))
    cdf(t, dog) = cf(TDist(dog), t)
    (distribution)->(μ;N;apply;cdf)
end
# ---- Uniform Dist ----
"""
    ## Uniform Distribution
    ### Description
      Calculates the Uniform distribution of an array.\n
      --------------------\n
    ### Input
      UniformDist(x)\n
      --------------------\n
      #### Positional Arguments
      Array{Any} - x:: Array for which the T distribution should use the
      data from.\n
      --------------------\n
     ### Output
     t:: A Lathe distribution\n
     ---------------------\n
     ### Functions
     Distribution.apply(xt) :: Applies the distribution to xt\n
     Distribution.cdf(xt) :: Applies the distribution's
     corresponding cummulitive distribution function.\n
     ---------------------\n
     ### Data
     μ :: Mean of the input data.\n
     N :: The length of the input data.
       """
function UniformDist(array)
    dist = Uniform(minimum(array), maximum(array))
    apply(xt) = pdf(dist, xt)
    cdf(x) = cdf(dist, x)
    (var) ->(dist;cdf;apply)
end
