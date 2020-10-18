#<---- correlation Coefficient --->
"""
    # Description
      A correlation coefficient is a statistical value that can be used to
      determine statistical correlation.\n
      --------------------\n
    # Input
      correlationcoeff(x, y)\n
      --------------------\n
      ## Positional Arguments
      x:: An array of values representing the general population.\n
      y:: An array of values representing the sample population.\n
      --------------------\n
     # Output
     r:: The correlation coefficient.
       """
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
"""

    # Description
      The TwoTailed function takes a distribution, a sample, and a confidence
          level, and will return a P value reflecting the probability of
          statistical significance.\n
      --------------------\n
    # Input
      TwoTailed(Distribution, sample; c)\n
      --------------------\n
      ## Positional Arguments
      Distribution:: A Lathe Distribution. <- Lathe Distribution\n
      sample:: An array of values representing the sample that should be
       tested. <- Array{Any}\n
       ## Key-word Arguments\n
       c:: Level of confidence for a given test in decimal form of a percentage.
       <- Float64
      --------------------\n
     # Output
     P:: P value representing the probability of A <= P >= - A
       """
function TwoTailed(dist, sample; c = .95)
    a = 1 - c
    test_stat = dist.apply(sample)
    dof = dist.N - 1
    return(test_stat)
end
"""

    # Description
      The OneTailed test function takes a distribution, a sample, and a
          confidence
          level, and will return a P value reflecting the probability of
          statistical significance.\n
      --------------------\n
    # Input
      TwoTailed(Distribution, sample; c)\n
      --------------------\n
      ## Positional Arguments
      Distribution:: A Lathe Distribution. <- Lathe Distribution\n
      sample:: An array of values representing the sample that should be
       tested. <- Array{Any}\n
       ## Key-word Arguments\n
       c:: Level of confidence for a given test in decimal form of a percentage.
       <- Float64
      --------------------\n
     # Output
     P:: P value representing the probability of A <= P >= - A
       """
function OneTailed(dist, sample; c = .95)
    a = 1 - c
    test_stat = dist.apply(sample)
    dof = dist.N - 1
    return(test_stat)
end
