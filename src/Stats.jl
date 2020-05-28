#================
Stats
    Module
================#
"""
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
      | ~~~~~~~~~~ Inferential ~~~~~~~~~~~\n
      |_____stats.independent_t(sample,general)\n
      |_____stats.spearman(var1,var2)\n
      |_____stats.pearson(x,y)\n
      |_____stats.chisqu(array)\n
      |_____stats.sign(array)\n
      |_____stats.f_test(sample,general)\n
      | ~~~~~~~~~~ Bayesian ~~~~~~~~~~~\n
      |_____stats.bay_ther(p,a,b)\n
      |_____stats.cond_prob(p,a,b)\n
      | ~~~~~~~~~~ Distributions ~~~~~~~~~~~\n
      |====== validate ======\n
      |____________/ Metrics ___________\n
      |_____validate.mae(actual,pred)\n
      |_____validate.r2(actual,pred)\n
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
export chi_dist, bernoulli_dist, binomial_dist, normal_dist
#==
Inferential
==#
include("stats/inferential.jl")
export independent_t, f_test, correlationcoeff, paired_t, spearman
export pearson, chisq, wilcoxrs, wilcoxsr, sign, f_test
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
