#================
Stats
    Module
================#
"""
      |====== Lathe.stats ======\n
      | 0-0-0-0- Base -0-0-0-0\n
      |_____mean(array)\n
      |_____mode(array)\n
      |_____variance(array)\n
      |_____confiints(data,confidence = .95)\n
      |_____standarderror(array)\n
      |_____std(data)\n
      |_____ste(data)\n
      |_____z(array)\n
      |_____firstquar(array)\n
      |_____secondquar(array)\n
      |_____thirdquar(array)\n
      |\n
      | 0-0-0-0- Inferential -0-0-0-0\n
      |_____independent_t(sample,general)\n
      |_____spearman(var1,var2)\n
      |_____pearson(x,y)\n
      |_____chisqu(array)\n
      |_____sign(array)\n
      |_____f_test(sample,general)\n
      |_____correlationcoeff(x,y)\n
      |_____dog(sample,general)\n
      |\n
      | 0-0-0-0- Bayesian -0-0-0-0\n
      |_____bay_ther(p,a,b)\n
      |_____cond_prob(p,a,b)\n
      | 0-0-0-0- Distributions -0-0-0-0\n
      |\n
      |_____t_dist(sample,general)
      |_____binomial_dist(positives,size)
      |_____normal_dist(sample)
      |\n
      | 0-0-0-0- Validation -0-0-0-0\n
      |_____mae(actual,pred)\n
      |_____mse(actual,pred)
      |_____r2(actual,pred)\n

       """
module stats
#==
Base
==#
include("stats/statbase.jl")
export mean, ste, std, median, mode, variance, confints, ste
export std, firstquar, secondquar, thirdquar, getranks, z
#==
Distributions
==#
include("stats/distributions.jl")
export chi_dist, bernoulli_dist, binomial_dist, normal_dist, t_dist
#==
Inferential
==#
include("stats/inferential.jl")
export independent_t, f_test, correlationcoeff, paired_t, spearman
export pearson, chisq, wilcoxrs, wilcoxsr, sign, f_test, dog
#==
Bayesian
==#
include("stats/bayesian.jl")
export bay_ther, cond_prob
#==
Validate
==#
include("stats/validate.jl")
export mae, mse, r2, catacc
#==
Macros
==#
include("stats/stats_macros.jl")
export @mu, @sigma, @r, @t, @f, @-, @chi, @p, @acc
#---------------------------
end
