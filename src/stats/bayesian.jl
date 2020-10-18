#<----Bayes Theorem---->
#P = prob, A = prior, B = Evidence,
"""

    # Description
      Bayes theorem takes probability, prior observations, and evidence
      as percentages and will return a posterior.\n
      --------------------\n
    # Input
      bay_ther(p, a, b)\n
      --------------------\n
      ## Positional Arguments
      Float64 - p:: A percentage representing probability.\n
      Float64 - a:: A percentage representing prior. <- Float64\n
      Float64 - b:: A percentage representing evidence. <- Float64\n
      --------------------\n
     # Output
     P:: Posterior value.
       """
function bay_ther(p,a,b)
    psterior = (p*(b|a) * p*(a)) / (p*b)
    return(psterior)
end
"""
      Performs Bayesian Conditional Probability and returns probability.\n
      --------------------\n
      prob = .50\n
      prior = .20\n
      evidence = .30\n
      p = cond_prob(prob,prior,evidence)\n
       """
function cond_prob(p,a,b)
    psterior = bay_ther(p,a,b)
    cond = p*(a|b)
    return(cond)
end
