#===============================
<-----------Lathe.jl----------->
Programmed by Emmett Boudreau
    <emmett@emmettboudreau.com>
        <http://emmettboudreau.com>
MIT General Open Source License
    (V 3.0.0)
        Free for Modification and
        Redistribution
=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=|
         CONTRIBUTORS
=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=|
        ~ emmettgb
        ~ stefanches7
/><><><><><><><><><><><><><><><><\
Thank you for your forks!
<-----------Lathe.jl----------->
38d8eb38-e7b1-11e9-0012-376b6c802672
#[deps]
DataFrames.jl
Random.jl
================================#
@doc """
      |====== Lathe - Easily ML =====\n
      |= = = = = v. 0.0.9 = = = = = |\n
      |==============================\n
      |__________Lathe.stats\n
      |__________Lathe.validate\n
      |__________Lathe.preprocess\n
      |__________Lathe.models\n
      |______________________________\n
      Use ?(Lathe.package) for information!\n
      [uuid]\n
      38d8eb38-e7b1-11e9-0012-376b6c802672\n
      [deps]\n
      DataFrames.jl\n
      Random.jl\n
       """ ->
module Lathe
# <------- PARTS ----->
# <------- PARTS ----->
# <------- DEPS ----->
using DataFrames
using Random
# <------- DEPS ----->
#================
Stats
    Module
================#
@doc """
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
      |_____stats.getranks(array,rev = false)\n
      | ~~~~~~~~~~ Inferential ~~~~~~~~~~~\n
      |_____stats.independent_t(sample,general)\n
      |_____stats.paired_t(array)\n
      |_____stats.spearman(var1,var2)\n
      |_____stats.pearson(x,y)\n
      |_____stats.chisqu(array)\n
      |_____stats.sign(array)\n
      |_____stats.f_test(sample,general)\n
      |_____stats.anova(arra)\n
      | ~~~~~~~~~~ Bayesian ~~~~~~~~~~~\n
      |_____stats.bay_ther(p,a,b)\n
      |_____stats.cond_prob(p,a,b)\n
      | ~~~~~~~~~~ Distributions ~~~~~~~~~~~\n
      |_____stats.bournelli_dist(array)\n
      |_____stats.binomial_dist(positives,size)\n
       """ ->
module stats
#<----Mean---->
@doc """
      Calculates the mean of a given array.\n
      --------------------\n
      array = [5,10,15]\n
      mean = Lathe.stats.mean(array)\n
      println(mean)\n
        10
       """ ->
function mean(array)
    observations = length(array)
    average = sum(array)/observations
    return(average)
end
#<----Median---->
@doc """
      Calculates the median (numerical center) of a given array.\n
      --------------------\n
      array = [5,10,15]\n
      median = Lathe.stats.median(array)\n
      println(median)\n
        10
       """
function median(array)
    n = length(array)
    half = n / 2
    current = 1
    sorted = sort!(array,rev = false)
    median = 0
    for i in sorted
        if current >= half
            median = i
        else
        current = current + 1
    end
    end
    return(median)
end
#<----Mode---->
@doc """
      Gives the digit most common in a given array\n
      --------------------\n
      array = [5,10,15,15,10,5,10]\n
      mode = Lathe.stats.mode(array)\n
      println(mode)\n
        10
       """ ->
function mode(array)
    m = findmax(array)
    return(m)
end
#<----Variance---->
@doc """
      Gives the variance of an array..\n
      --------------------\n
      array = [5,10,15]\n
      variance = Lathe.stats.variance(array)\n
       """ ->
function variance(array)
    me = mean(array)
    sq = sum(array) - me
    squared_mean = sq ^ 2
    return(squared_mean)
end
#<----Confidence Intervals---->
@doc """
      Returns the confidence intervals of given data.\n
      --------------------\n
      array = [5,10,15]\n
      confidence = .98\n
      low, high = Lathe.stats.confints(array,confidence)\n
      --------------------\n
      PARAMETERS\n
      confidence: Confidence is a float percentage representing the level of
      confidence required in your test. The confidence metric is used
      exclusively for calculating the interval.
       """ ->
function confiints(data, confidence=.95)
    mean = mean(data)
    std = std(data)
    stderr = standarderror(data)
#    interval = stderr * scs.t.ppf((1 + confidence) / 2.0, n-1)
#    return (mean-interval, mean+interval)
end
#<----Standard Error---->
@doc """
      Calculates the Standard Error of an array.\n
      --------------------\n
      array = [5,10,15]\n
      ste = Lathe.stats.standarderror(array)\n
       """ ->
function standarderror(data)
    std = std(data)
    sample = length(data)
    ste = (std/sqrt(sample))
    return(ste)
end
#<----Standard Deviation---->
@doc """
      Calculates the Standard Deviation of a given array.\n
      --------------------\n
      array = [5,10,15]\n
      std = Lathe.stats.std(array)\n
       """ ->
function std(array3)
    m = mean(array3)
    [i = (i-m) ^ 2 for i in array3]
    m = mean(array3)
    m = sqrt(Complex(m))
    return(m)
end
#<---- correlation Coefficient --->
@doc """
      Calculates the Correlation Coeffiecient of between two features\n
      --------------------\n
      x = [5,10,15]\n
      y = [5,10,15]\n
      r = Lathe.stats.correlationcoeff(x,y)\n
       """ ->
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
#<----Z score---->
@doc """
      Calculates the Z score of a given array.\n
      --------------------\n
      array = [5,10,15]\n
      z = Lathe.stats.z(array)\n
       """ ->
function z(array)
    x̄ = mean(array)
    σ = std(array)
    return map(x -> (x - x̄) / σ, array)
end
#<----Quartiles---->
# - First
@doc """
      Returns the point in an array located at 25 percent of the sorted data.\n
      --------------------\n
      array = [5,10,15]\n
      q1 = Lathe.stats.firstquar(array)\n
       """ ->
function firstquar(array)
    m = median(array)
    q1 = array * .5
    return(q1)
end
# - Second(median)
@doc """
      Returns the point in an array located at 50 percent of the sorted data.
      The second quartile is also known as the median, or the middle of the sorted data.\n
      --------------------\n
      array = [5,10,15]\n
      q2 = Lathe.stats.secondquar(array)\n
       """ ->
function secondquar(array)
    m = median(array)
    return(m)
end
# - Third
@doc """
      Returns the point in an array located at 75 percent of the sorted data.\n
      --------------------\n
      array = [5,10,15]\n
      q2 = Lathe.stats.secondquar(array)\n
       """ ->
function thirdquar(array)
    q = median(array)
    q3 = q * 1.5
    return(q3)
end
# <---- Rank ---->
@doc """
      Ranks indices in an array based on quantitative weight (count of the
      numbers) and returns a new array of the ranks of each column. This
      function is made primarily for the Wilcox Rank-Sum test.\n
      --------------------\n
      array = [5,10,15]\n
      q2 = Lathe.stats.secondquar(array)\n
       """ ->
function getranks(array,rev = false)
    sortedar = sort(array,rev=rev)
    num = 1
    list = []
    for i in sortedar
        append!(list,i)
        num = num + 1
    end
    return(list)
end
#-------Inferential-----------__________
#<----T Test---->
# - Independent
@doc """
      Performs an independent T test.\n
      --------------------\n
      sample = [5,10,15]
      general = [15,25,35]\n
      t = Lathe.stats.independent_t(sample,general)\n
       """ ->
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
@doc """
    THIS FUNCTION IS NOT YET WRITTEN\n
      Paired T (Dependent T) is a T-test that doesn't require a sample.\n
      --------------------\n
      array = [5,10,15]\n
      q2 = Lathe.stats.paired_t(var1,var2)\n
       """ ->
function paired_t(var1,var2)
    d = var1 .- var2
    d̄ = mean(x)
end
#<---- Correlations ---->
# - Spearman
@doc """
      Returns a probability using a Spearman correlation.\n
      --------------------\n
      var1 = [5,10,15]\n
      var2 = [5,4,3,5,6]\n
      r = Lathe.stats.spearman(var1,var2)\n
       """ ->
function spearman(var1,var2)
    rgX = getranks(var1)
    rgY = getranks(var2)
    ρ = rgX*rgY / (std(rgX)*std(rgY))
    return(ρ)
end
# - Pearson
@doc """
      Returns a probability using a Pearson correlation.\n
      --------------------\n
      x = [5,10,15]\n
      y = [5,4,3,5,6]\n
      r = Lathe.stats.spearman(x,y)\n
       """ ->
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
@doc """
      Returns a probability using a chi squared distribution.\n
      --------------------\n
      var1 = [5,10,15]\n
      var2 = [5,4,3,5,6]\n
      p = Lathe.stats.chisq(var1,var2)\n
       """ ->
function chisq(var1,var2)
    chistat(obs, exp) = (obs - exp)^2/exp
    return chistat.(x, e) |> sum
end
#<---- ANOVA ---->
@doc """
      FUNCTION NOT YET WRITTEN\n
      Anova is used to analyze variance in an array\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """ ->
function anova(var1,var2)

end
#<---- Wilcoxon ---->
# - Wilcoxon Rank-Sum Test
function wilcoxrs(var1,var2)
    #Hash
end
@doc """
      FUNCTION NOT YET WRITTEN\n
      Wilcox Sum Rank Tests are used to determine a probability with ranks\n
      --------------------\n
      var1 = [5,10,15]\n
      var2 = [19,25,30]\n
      p = Lathe.stats.wilcoxsr(var1,var2)\n
       """ ->
function wilcoxsr(var1,var2)
    #Hash
end
#<---- Sign Test ---->
@doc """
      The Sign test determines correlation through negative and positive
      placement with binomial distribution.\n
      --------------------\n
      var1 = [5,10,15]\n
      var2 = [19,25,30]\n
      p = Lathe.stats.sign(var1,var2)\n
       """ ->
function sign(var1,var2)
    sets = var1 .- var2
    positives = []
    negatives = []
    zeros = []
    for i in sets
        if i == 0
            zeros.append(i)
        elseif i > 0
            positives.append(i)
        elseif i < 0
            negatives.append(i)
        end
    end
    totalpos = length(positives)
    totallen = length(sets)
    ans = binomial_dist(positives,totallen)
    return(ans)
end
#<---- F-Test---->
@doc """
      An F test returns a probability of correlation, and is used similarly
      to a T test.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """ ->
function f_test(sample,general)
    totvariance = variance(general)
    sampvar = variance(sample)
    f =  sampvar / totvariance
    return(f)
end
#-------Bayesian--------------___________
#<----Bayes Theorem---->
#P = prob, A = prior, B = Evidence,
function bay_ther(p,a,b)
    psterior = (p*(b|a) * p*(a)) / (p*b)
    return(psterior)
end
@doc """
      Performs Bayesian Conditional Probability and returns probability.\n
      --------------------\n
      prob = .50\n
      prior = .20\n
      evidence = .30\n
      p = Lathe.stats.cond_prob(prob,prior,evidence)\n
       """ ->
function cond_prob(p,a,b)
    psterior = bay_ther(p,a,b)
    cond = p*(a|b)
    return(cond)
end
#=========================
Distributions section!!!!!
~Added Lathe 0.0.6 ~
=========================#
function bernoulli_dist()
    # P(x) = P^x(1-P)^1-x for x=0 eller 1
end
@doc """
      Binomial Distribution is a distribution well known for its use in
           statistical tests and decision making models.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """ ->
function binomial_dist(positives,size)
    # p = n! / x!(n-x!)*π^x*(1-π)^N-x
    n = size
    x = positives
    factn = factorial(big(n))
    factx = factorial(big(x))
    nx = factn / (factx * (n-x))
    return(nx)
end
# <---- Chi Distribution --->
@doc """
      FUNCTION NOT YET WRITTEN\n
      Chi Distribution in another well-known distribution well known for being
      used in statistical tests.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """ ->
function chidist(x,e)
    #
end
#---------------------------
end

#================
Preprocessing
     Module
================#
@doc """
      |====== Lathe.preprocess =====\n
      |____________/ Generalized Processing ___________\n
      |_____preprocess.TrainTestSplit(array)\n
      |_____preprocess.ArraySplit(array)\n
      |_____preprocess.SortSplit(array)\n
      |_____preprocess.UniformSplit(array)\n
      |____________/ Feature Scaling ___________\n
      |_____preprocess.Rescalar(array)\n
      |_____preprocess.ArbitraryRescale(array)\n
      |_____preprocess.MeanNormalization(array)\n
      |_____preprocess.StandardScalar(array)\n
      |_____preprocess.UnitLScale(array)\n
      |____________/ Categorical Encoding ___________\n
      |_____preprocess.OneHotEncode(array)\n
      |_____preprocess.InvertEncode(array)\n

       """ ->
module preprocess
using Random
using Lathe
#===============
Generalized
    Data
        Processing
===============#
# Train-Test-Split-----
@doc """
      Train Test split is used to create a validation set to toy accuracy
      with. TrainTestSplit() takes a DataFrame and splits it at a certain
      percentage of the data.\n
      --------------------\n
      df = DataFrame(:A => [1,2,3],:B => [4,5,6])\n
      test,train = Lathe.preprocess.TrainTestSplit(df,at = 0.75)\n
      -------------------\n
      PARAMETERS:\n
      at:: Percentage value used to determine a point to split the data.
       """ ->
function TrainTestSplit(df,at = 0.75)
    sample = randsubseq(1:size(df,1), at)
    trainingset = df[sample, :]
    notsample = [i for i in 1:size(df,1) if isempty(searchsorted(sample, i))]
    testset = df[notsample, :]
    return(trainingset,testset)
end
# Array-Split ----------
@doc """
      Array Split does the exact same thing as TrainTestSplit(), but to an
      an array instead of a DataFrame\n
      --------------------\n
      array = [5,10,15]\n
      test, train = Lathe.preprocess.ArraySplit(array,at = 0.75)\n
      -------------------\n
      PARAMETERS:\n
      at:: Percentage value used to determine a point to split the data.
       """ ->
function ArraySplit(data, at = 0.7)
    n = length(data)
    idx = Random.shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
    return(test_idx,train_idx)
end
# Sort-Split -------------
@doc """
      SortSplit sorts the data from least to greatest, and then splits it,
      ideal for quartile calculations.\n
      --------------------\n
      array = [5,10,15]\n
      top25, lower75 = Lathe.preprocess.SortSplit(array,at = 0.75,rev = false)\n
      -------------------\n
      PARAMETERS:\n
      at:: Percentage value used to determine a point to split the data.\n
      rev:: Reverse, false by default, determines whether to sort least to
      greatest, or greatest to least.\n
       """ ->
function SortSplit(data, at = 0.25, rev=false)
  n = length(data)
  sort!(data, rev=rev)  # Sort in-place
  train_idx = view(data, 1:floor(Int, at*n))
  test_idx = view(data, (floor(Int, at*n)+1):n)
  return(test_idx,train_idx)
end
# Unshuffled Split ----
@doc """
      Uniform Split does the exact same thing as ArraySplit(), but observations
      are returned split, but unsorted and unshuffled.\n
      --------------------\n
      array = [5,10,15]\n
      test, train = Lathe.preprocess.UniformSplit(array,at = 0.75)\n
      -------------------\n
      PARAMETERS:\n
      at:: Percentage value used to determine a point to split the data.
       """ ->
function UniformSplit(data, at = 0.7)
    n = length(data)
    idx = data
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
    return(test_idx,train_idx)
end
#=======
Numerical
    Scaling
=======#
# ---- Rescalar (Standard Deviation) ---
@doc """
      Rescalar scales a feature based on the minimum and maximum of the array.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.Rescalar(array)\n
       """ ->
function Rescalar(array)
    min = minimum(array)
    max = maximum(array)
    v = [i = (i-min) / (max - min) for i in array]
    return(v)
end
# ---- Arbitrary Rescalar ----
@doc """
      Arbitrary Rescaling scales a feature based on the minimum and maximum
       of the array.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.Rescalar(array)\n
       """ ->
function ArbitraryRescale(array)
    a = minimum(array)
    b = maximum(array)
    v = [x = a + ((i-a*i)*(b-a)) / (b-a) for x in array]
    return(v)
end
# ---- Mean Normalization ----
@doc """
      Mean Normalization normalizes the data based on the mean.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.MeanNormalization(array)\n
       """ ->
function MeanNormalization(array)
    avg = Lathe.stats.mean(array)
    a = minimum(array)
    b = maximum(array)
    v = [i = (i-avg) / (b-a) for i in array]
    return(v)
end
# ---- Quartile Normalization ----
function QuartileNormalization(array)
    q1 = firstquar(array)
    q2 = thirdquar(array)

end
# ---- Z Normalization ----
@doc """
      Standard Scalar z-score normalizes a feature.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.StandardScalar(array)\n
       """ ->
function StandardScalar(array)
    q = Lathe.stats.std(array)
    avg = Lathe.stats.mean(array)
    v = [i = (i-avg) / q for i in array]
    return(v)
end
# ---- Unit L-Scale normalize ----
@doc """
      FUNCTION NOT YET WRITTEN\n
      Unit L Scaling uses eigen values to normalize the data.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.UnitLScale(array)\n
       """ ->
function UnitLScale(array)

end
#==========
Categorical
    Encoding
==========#
# <---- One Hot Encoder ---->

@doc """
      One hot encoder replaces a single feature with sub arrays containing
      boolean values (1 or 0) for each individual category.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.OneHotEncode(array)\n
       """ ->
function OneHotEncode(array)
    # define a mapping of chars to integers
#    char_to_int = dict((c, i) for i, c in enumerate(array))
#    int_to_char = dict((i, c) for i, c in enumerate(array))
    # integer encode input data
#    integer_encoded = [char_to_int[char] for char in data]
    # one hot encode
#    onehot_encoded = []
#    for value in integer_encoded
#    	letter = [0 for _ in 0:len(alphabet)]
#    	letter[value] = 1
#    	append!(oonehot_encoded,letter)
#     end
#    return(onehot_encoded)
end
# <---- Invert Encoder ---->
#==
@doc """
      FUNCTION NOT YET WRITTEN\n
      Invert Encoder (Not written.)\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.OneHotEncode(array)\n
       """ ->
       ==#
function InvertEncode(array)

end
#-----------------------------
end
#================
Predictive
    Learning
        Models
================#
@doc """
      |====== Lathe.models =====\n
      |____________/ Accessories ___________\n
      |_____models.predict(m,xt)\n
      |_____models.Pipeline([steps],model)\n
      |____________/ Continuous models ___________\n
      |_____models.meanBaseline(y)\n
      |_____models.RegressionTree(x,y,n_divisions)\n
      |_____models.FourSquare(x,y)\n
      |_____models.IsotonicRegression(x,y)\n
      |_____models.MultipleLinearRegression([x],y)\n
      |_____models.RidgeRegression(x,y)\n
      |_____models.LinearRegression(x,y)\n
      |_____models.LinearLeastSquare(x,y,Type)\n
      |____________/ Categorical Models ___________\n
      |_____models.LogisticRegression(x,y)\n
      |_____models.majBaseline\n
       """ ->
module models
#==
Base
    Models
        Functions
==#
using Lathe
using Random
#===========
Accessories
===========#
@doc """
      Pipelines can contain a predictable Lathe model with preprocessing that
      occurs automatically. This is done by putting X array processing methods
      into the iterable steps, and then putting your Lathe model in.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      model = Lathe.models.meanBaseline(y)\n
      StandardScalar = Lathe.preprocess.StandardScalar\n
      MeanNormalization = Lathe.preprocess.MeanNormalization\n
      steps = [StandardScalar,MeanNormalization]\n
      pipeline = Lathe.models.Pipeline(steps,model)\n
      y_pred = Lathe.models.predict(pipeline,xtrain)\n
      --------------------\n
      HYPER PARAMETERS\n
      steps:: Iterable list (important, use []) of processing methods to be
      performed on the xtrain set. Note that it will not be applied to the
      train set, so preprocessing for the train set should be done before
      model construction.\n
      model:: Takes any Lathe model, uses Lathe.models.predict,\n
      method assersion is still do-able with the dispatch, meaning any model\n
      designed to work with Lathe.models (and Lathe.models.predict) will work\n
      inside of a Lathe pipeline."""
mutable struct Pipeline
    steps
    model
end
function pred_pipeline(m,x)
    for step in m.steps
        x = step(x)
    end
    x = [x = step(x) for step in m.steps]
    ypr = Lathe.models.predict(m.model,x)
    return(ypr)
end
#==============
========================================================
=======================================================================
            CONTINUOS MODELS               CONTINUOS MODELS
            CONTINUOS MODELS               CONTINUOS MODELS
======================================================================
======================================================================#
#==
Mean
    Baseline
==#
 # Model Type
 @doc """
       A mean baseline is great for getting a basic accuracy score in order
           to make a valid direction for your model.\n
       --------------------\n
       x = [7,6,5,6,5]\n
       y  = [3.4.5.6.3]\n
       xtrain = [7,5,4,5,3,5,7,8]\n
       model = Lathe.models.meanBaseline(y)
       y_pred = Lathe.models.predict(model,xtrain)\n
        """ ->
mutable struct meanBaseline
    y
end
#----  Callback
function pred_meanbaseline(m,xt)
    e = []
    m = Lathe.stats.mean(m.y)
    print("-Lathe.models Mean Baseline-")
    print("mean: ",m)
    for i in xt
        append!(e,m)
    end
    return(e)
end
#==
Regression
    Tree
==#
# Model Type
@doc """
      Pipelines can contain a predictable Lathe model with preprocessing that
      occurs automatically.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      n_divisions = 4\n
      model = Lathe.models.RegressionTree(x,y,n_divisions)\n
      --------------------\n
      HYPER PARAMETERS\n
      n_divisions:: n_divisions determines the number of divisions that the
      regression tree should take."""
mutable struct RegressionTree
    x
    y
    n_divisions
end
#----  Callback
function pred_regressiontree(m,xt)

end
#==
Four
    Square
==#
@doc """
      A FourSquare splits data into four linear least squares, and then
      predicts variables depending on their location in the data (in
      quartile range.) With the corresponding model for said quartile.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      model = Lathe.models.FourSquare(x,y)\n"""
mutable struct FourSquare
    x
    y
end
#----  Callback
function pred_foursquare(m,xt)
    # x = q1(r(floor:q1)) |x2 = q2(r(q1:μ)) |x3 = q3(r(q2:q3)) |x4 q4(r(q3:cieling))
    # y' = q1(x * (a / x)) | μ(x * (a / x2)) | q3(x * (a / x3) | q4(x * (a / x4))
        x = m.x
        y = m.y
        # Go ahead and throw an error for the wrong input shape:
        xlength = length(x)
        ylength = length(y)
        if xlength != ylength
            throw(ArgumentError("The array shape does not match!"))
        end
        # Our empty Y prediction list==
        e = []
        # Quad Splitting the data ---->
        # Split the Y
        y2,range1 = Lathe.preprocess.SortSplit(y)
        y3,range2 = Lathe.preprocess.SortSplit(y2)
        y4,range3 = Lathe.preprocess.SortSplit(y3)
        y5,range4 = Lathe.preprocess.SortSplit(y4)
        yrange5 = y5
        # Split the x train
        x1,xrange1 = Lathe.preprocess.SortSplit(x)
        x2,xrange2 = Lathe.preprocess.SortSplit(x1)
        x3,xrange3 = Lathe.preprocess.SortSplit(x2)
        x4,xrange4 = Lathe.preprocess.SortSplit(x3)
        xrange5 = y5
        # Fitting the 4 linear regression models ---->
        regone = LinearLeastSquare(xrange1,range1, :REG)
        regtwo = LinearLeastSquare(xrange2,range2, :REG)
        regthree = LinearLeastSquare(xrange3,range3, :REG)
        regfour = LinearLeastSquare(xrange4,range4, :REG)
        regfive = LinearLeastSquare(xrange5,yrange5, :REG)
        # Split the train Data
        xt1,xtrange1 = Lathe.preprocess.SortSplit(xt)
        xt2,xtrange2 = Lathe.preprocess.SortSplit(xt1)
        xt3,xtrange3 = Lathe.preprocess.SortSplit(xt2)
        xt4,xtrange4 = Lathe.preprocess.SortSplit(xt3)
        xtrange5 = xt4
        # Get min-max
        xtrange1min = minimum(xtrange1)
        xtrange1max = maximum(xtrange1)
        xtrange2min = minimum(xtrange2)
        xtrange2max = maximum(xtrange2)
        xtrange3min = minimum(xtrange3)
        xtrange3max = maximum(xtrange3)
        xtrange4min = minimum(xtrange4)
        xtrange4max = maximum(xtrange4)
        xtrange5min = minimum(xtrange5)
        # Ranges for ifs
        condrange1 = (xtrange1min:xtrange1max)
        condrange2 = (xtrange2min:xtrange2max)
        condrange3 = (xtrange3min:xtrange3max)
        condrange4 = (xtrange4min:xtrange4max)
        # This for loop is where the dimension's are actually used:
        for i in xt
            if i in condrange1
                ypred = predict(regone,i)
            elseif i in condrange2
                ypred = predict(regtwo,i)
            elseif i in condrange3
                ypred = predict(regthree,i)
            elseif i in condrange4
                ypred = predict(regfour,i)
            else
                ypred = predict(regfive,i)
            end
            append!(e,ypred)
        end
        return(e)
end
#==
Isotonic
    Regression
==#
@doc """
      FUNCTION NOT YET WRITTEN\n
      Isotonic Regression is used to predict continuous features with high
      variance.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.OneHotEncode(array)\n"""
mutable struct IsotonicRegression
    x
    y
end
function pred_isotonicregression(m,xt)
    if length(m.x) != length(m.y)
        throw(ArgumentError("The array shape does not match!"))
    end
end
#==
Multiple
    Linear
        Regression
==#
@doc """
      Multiple Linear Regression is used to influence LinearRegression with
      multiple features by averaging their predictions.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      model = Lathe.models.MultipleLinearRegression(x,y)\n"""
mutable struct MultipleLinearRegression
    x
    y
end
function pred_multiplelinearregression(m,xt)
    if length(m.x) != length(xt)
        throw(ArgumentError("Bad Feature Shape |
        Training Features are not equal!",))
    end
    y_pred = []
    for z in xt
        r = 0
        predavg = []
        for i in z
            m = LinearRegression(z,m.y)
            pred = predict(m,z)
            append!(predavg,pred)
        end
        append!(y_pred,predavg)
    end
    len = length(y_pred[1])
    yprl = length(y_pred)
    pr = []
    numbers = collect(1:yprl)
    oddsonly = numbers[numbers .% 2 .== 0]
    oddsonly = filter!(e->e≠0,oddsonly)
    if yprl in oddsonly
        truonly = true
    else
        truonly = false
    end
        for z in oddsonly
            cp = z + 1
            for i in 1:len
                s = z[i]
                v = cp[i]
                d = Lathe.stats.mean([s,v])
                append!(pr,d)
            end
        end
        if truonly == true
            fn = maximum(oddsonly)
            z = fn
            cp = fn + 1
            for i in 1:len
                s = z[i]
                v = cp[i]
                d = Lathe.stats.mean([s,v])
                append!(pr,d)
            end
        end
    return(pr)
end
#==
Linear
    Regression
==#

@doc """
      Linear Regression is a well-known linear function used for predicting
      continuous features with a mostly linear or semi-linear slope.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      model = Lathe.models.LinearRegression(x,y)
      y_pred = Lathe.models.predict(model,xtrain)\n
       """
mutable struct LinearRegression
    x
    y
end
#----  Callback
function pred_LinearRegression(m,xt)
    # a = ((∑y)(∑x^2)-(∑x)(∑xy)) / (n(∑x^2) - (∑x)^2)
    # b = (x(∑xy) - (∑x)(∑y)) / n(∑x^2) - (∑x)^2
    if length(m.x) != length(m.y)
        throw(ArgumentError("The array shape does not match!"))
    end
    # Get our x and y as easier variables
    x = m.x
    y = m.y
    # Get our Summations:
    Σx = sum(x)
    Σy = sum(y)
    # dot x and y
    xy = x .* y
    # ∑dot x and y
    Σxy = sum(xy)
    # dotsquare x
    x2 = x .^ 2
    # ∑ dotsquare x
    Σx2 = sum(x2)
    # n = sample size
    n = length(x)
    # Calculate a
    a = (((Σy) * (Σx2)) - ((Σx * (Σxy)))) / ((n * (Σx2))-(Σx^2))
    # Calculate b
    b = ((n*(Σxy)) - (Σx * Σy)) / ((n * (Σx2)) - (Σx ^ 2))
    xt = [i = a + (b * i) for i in xt]
    return(xt   )
end
#==
Linear
    Least
     Square
==#
@doc """
      Least Squares is ideal for predicting continous features.
      Many models use Least Squares as a base to build off of.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      Type = :LIN
      model = Lathe.models.LeastSquare(x,y,Type)\n
      y_pred = Lathe.models.predict(model,xtrain)\n
      -------------------\n
      HYPER PARAMETERS\n
      Type:: Type determines which Linear Least Square algorithm to use,
      :LIN, :OLS, :WLS, and :GLS are the three options.\n
      - :LIN = Linear Least Square Regression\n
      - :OLS = Ordinary Least Squares\n
      - :WLS = Weighted Least Squares\n
      - :GLS = General Least Squares
       """
mutable struct LeastSquare
    x
    y
    Type
end
function pred_leastsquare(m,xt)
    if length(m.x) != length(m.y)
        throw(ArgumentError("The array shape does not match!"))
    end
    if m.Type == :LIN
        x = m.x
        y = m.y
        xy = x .* y
        sxy = sum(xy)
        n = length(x)
        x2 = x .^ 2
        sx2 = sum(x2)
        sx = sum(x)
        sy = sum(y)
        # Calculate the slope:
        slope = ((n*sxy) - (sx * sy)) / ((n * sx2) - (sx)^2)
        # Calculate the y intercept
        b = (sy - (slope*sx)) / n
        y_pred = [x = (slope * x) + b for x in xt]
    end
    if m.Type == :OLS

    end
    if m.Type == :WLS

    end
    if m.Type == :GLS

    end
    if m.Type == :GRG

    end
    return(y_pred)
end
#==
Ridge
    Regression
==#
@doc """
      Ridge Regression is another regressor ideal for predicting linear,
          continuous features.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.OneHotEncode(array)\n
       """
mutable struct RidgeRegression
    x
    y
end
function pred_ridgeregression(m,xt)
    if length(m.x) != length(m.y)
        throw(ArgumentError("The array shape does not match!"))
    end
end

#======================================================================
=======================================================================
            CATEGORICAL MODELS             CATEGORICAL MODELS
            CATEGORICAL MODELS             CATEGORICAL MODELS
======================================================================
======================================================================#
#==
Majority
    Class
        Baseline
==#
@doc """
      FUNCTION NOT YET WRITTEN\n
      Majority class baseline is used to find the most often interpreted
      classification in an array.\n
      --------------------\n
       """
mutable struct majBaseline
    y
end
#----  Callback
function pred_majbaseline(m,xt)
    y = m.y
    e = []
    mode = Lathe.stats.mode(xt)
    for i in xt
        append!(e,i)
    end

end
#==
Multinomial
    Naive
        Bayes
==#
mutable struct MultinomialNB
    x
    y
end
function pred_multinomialnb(m,xt)

end
#==
Logistic
    Regression
==#
#==
@doc """
      One hot encoder replaces a single feature with sub arrays containing
      boolean values (1 or 0) for each individual category.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.OneHotEncode(array)\n
       """ ->
       ==#
mutable struct LogisticRegression
    x
    y
end
function pred_logisticregression(m,xt)
    if length(m.x) != length(m.y)
        throw(ArgumentError("The array shape does not match!"))
    end

end
#=====
Prediction
    Dispatch
=====#
predict(m::meanBaseline,x) = pred_meanbaseline(m,x)
predict(m::FourSquare,x) = pred_foursquare(m,x)
predict(m::majBaseline,x) = pred_majbaseline(m,x)
predict(m::RegressionTree,x) = pred_regressiontree(m,x)
predict(m::LinearRegression,x) = pred_LinearRegression(m,x)
predict(m::RidgeRegression,x) = pred_ridgeregression(m,x)
predict(m::LeastSquare,x) = pred_leastsquare(m,x)
predict(m::LogisticRegression,x) = pred_logisticregression(m,x)
predict(m::Pipeline,x) = pred_pipeline(m,x)
#
#----------------------------------------------
end
#==
This is the end of the main
module, nothing is to be written
beyond here
==#
end
