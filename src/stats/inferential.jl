#<---- correlation Coefficient --->
"""
      Calculates the Correlation Coeffiecient of between two features\n
      --------------------\n
      x = [5,10,15]\n
      y = [5,10,15]\n
      r = correlationcoeff(x,y)\n
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
#<---- Sign Test ---->
"""
      The Sign test determines correlation through negative and positive
      placement with binomial distribution.\n
      --------------------\n
      var1 = [5,10,15]\n
      var2 = [19,25,30]\n
      p = sign(var1,var2)\n
       """
function sign(var1,var2)
    sets = var1 .- var2
    positives = []
    negatives = []
    zeros = []
    for i in sets
        if i == 0
            append!(zeros,i)
        elseif i > 0
            append!(positives,i)
        elseif i < 0
            append!(negatives,i)
        end
    end
    totalpos = length(positives)
    totallen = length(sets)
    ans = binomial_dist(totalpos,totallen)
    return(ans)
end
# These two tests are incomplete, and need to call
#    the cummulative functions for their respective distributions.
# (Distribution.cdf)
function TwoTailed(dist, sample; c = .95)
    a = 1 - c
    test_stat = dist.apply(sample)
    dof = dist.N - 1
    return(test_stat)
end
function OneTailed(dist, sample; c = .95)
    a = 1 - c
    test_stat = dist.apply(sample)
    dof = dist.N - 1
    return(test_stat)
end
