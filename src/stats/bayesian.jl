#<----Bayes Theorem---->
#P = prob, A = prior, B = Evidence,
"""
      Performs Bayes Theorem and returns probability.\n
      --------------------\n
      prob = .50\n
      prior = .20\n
      evidence = .30\n
      p = bay_ther(prob,prior,evidence)\n
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
