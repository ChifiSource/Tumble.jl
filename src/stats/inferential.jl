#<---- correlation Coefficient --->
"""
      Calculates the Correlation Coeffiecient of between two features\n
      --------------------\n
      x = [5,10,15]\n
      y = [5,10,15]\n
      r = Lathe.stats.correlationcoeff(x,y)\n
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
#<----T Test---->
# - Independent
"""
      Performs an independent T test.\n
      --------------------\n
      sample = [5,10,15]
      general = [15,25,35]\n
      t = Lathe.stats.independent_t(sample,general)\n
       """
function independent_t(sample,general)
    sampmean = mean(sample)
    genmean = mean(general)
    samples = length(sample)
    m = genmean
    [i = (i-m) ^ 2 for i in general]
    m = mean(general)
    m = sqrt(m)
    std = m
    t = (sampmean - genmean) / (std / sqrt(samples))
    return(t)
end
# - Paired
"""
    THIS FUNCTION IS NOT YET WRITTEN\n
      Paired T (Dependent T) is a T-test that doesn't require a sample.\n
      --------------------\n
      array = [5,10,15]\n
      q2 = Lathe.stats.paired_t(var1,var2)\n
       """
function paired_t(var1,var2)
    d = var1 .- var2
    d̄ = mean(x)
end
#<---- Correlations ---->
# - Spearman
"""
      Returns a probability using a Spearman correlation.\n
      --------------------\n
      var1 = [5,10,15]\n
      var2 = [5,4,3,5,6]\n
      r = Lathe.stats.spearman(var1,var2)\n
       """
function spearman(var1,var2)
    rgX = getranks(var1)
    rgY = getranks(var2)
    ρ = rgX*rgY / (std(rgX)*std(rgY))
    return(ρ)
end
# - Pearson
"""
      Returns a probability using a Pearson correlation.\n
      --------------------\n
      x = [5,10,15]\n
      y = [5,4,3,5,6]\n
      r = Lathe.stats.spearman(x,y)\n
       """
function pearson(x,y)
    sx = std(x)
    sy = std(y)
    x̄ = mean(x)
    ȳ = mean(x)
    [i = (i-x̄) / sx for i in x]
    [i = (i-ȳ) / sy for i in y]
    n1 = n-1
    mult = x .* y
    sq = sum(mult)
    corrcoff = sq / n1
    return(corrcoff)
end
#<---- Chi-Square ---->
"""
      Returns a probability using a chi squared distribution.\n
      --------------------\n
      var1 = [5,10,15]\n
      var2 = [5,4,3,5,6]\n
      p = Lathe.stats.chisq(var1,var2)\n
       """
function chisq(var1,var2)
    chistat(obs, exp) = (obs - exp)^2/exp
    return chistat.(x, e) |> sum
end
#<---- Wilcoxon ---->
# - Wilcoxon Rank-Sum Test
"""
      FUNCTION NOT YET WRITTEN\n
      Wilcox Sum Rank Tests are used to determine a probability with ranks\n
      --------------------\n
      var1 = [5,10,15]\n
      var2 = [19,25,30]\n
      p = Lathe.stats.wilcoxsr(var1,var2)\n
       """
function wilcoxrs(var1,var2)
    #Hash
end
"""
      FUNCTION NOT YET WRITTEN\n
      Wilcox Sum Rank Tests are used to determine a probability with ranks\n
      --------------------\n
      var1 = [5,10,15]\n
      var2 = [19,25,30]\n
      p = Lathe.stats.wilcoxsr(var1,var2)\n
       """
function wilcoxsr(var1,var2)
    #Hash
end
#<---- Sign Test ---->
"""
      The Sign test determines correlation through negative and positive
      placement with binomial distribution.\n
      --------------------\n
      var1 = [5,10,15]\n
      var2 = [19,25,30]\n
      p = Lathe.stats.sign(var1,var2)\n
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
#<---- F-Test---->
@doc """
      An F test returns a probability of correlation, and is used similarly
      to a T test.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """
function f_test(sample,general)
    totvariance = variance(general)
    sampvar = variance(sample)
    f =  sampvar / totvariance
    return(f)
end
