#================
Preprocessing
     Module
================#
@doc """
      |====== Lathe.preprocess =====\n
      |____________/ Generalized Processing ___________\n
      |_____preprocess.TrainTestSplit(array)\n
      |_____preprocess.SortSplit(array)\n
      |_____preprocess.UniformSplit(array)\n
      |____________/ Feature Scaling ___________\n
      |_____preprocess.Rescalar(array)\n
      |_____preprocess.ArbitraryRescale(array)\n
      |_____preprocess.MeanNormalization(array)\n
      |_____preprocess.StandardScalar(array)\n
      |____________/ Categorical Encoding ___________\n
      |_____preprocess.OneHotEncode(array)\n

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

function _dfTrainTestSplit(df,at = 0.75)
    sample = randsubseq(1:size(df,1), at)
    trainingset = df[sample, :]
    notsample = [i for i in 1:size(df,1) if isempty(searchsorted(sample, i))]
    testset = df[notsample, :]
    return(trainingset,testset)
end
function _ArraySplit(data, at = 0.7)
    n = length(data)
    idx = Random.shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
    return(test_idx,train_idx)
end
@doc """
      TrainTestSplit takes either a DataFrame or an Array and splits it according to the at parameter.\n
      --------------------\n
      [data] <- Iterable dictionary, dataframe, or Array.\n
      a <- Percentage value used to determine a point to split the data.\n
      -------------------\n
       """
TrainTestSplit(data::Array, at::Float64) = _ArraySplit(data,at)
TrainTestSplit(data::DataFrame, at::Float64) = dfTrainTestSplit(data,at)
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
       """
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
       """
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
      --------------------\n
      ==Functions==\n
      predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)
       """ ->
function Rescalar(array)
    min = minimum(array)
    max = maximum(array)
    predict(array) = [i = (i-min) / (max - min) for i in array]
    (var) -> (predict)
end
# ---- Arbitrary Rescalar ----
@doc """
      Arbitrary Rescaling scales a feature based on the minimum and maximum
       of the array.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.Rescalar(array)\n
      --------------------\n
      ==Functions==\n
      predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)
       """ ->
function ArbitraryRescale(array)
    a = minimum(array)
    b = maximum(array)
    predict(array) = [x = a + ((i-a*i)*(b-a)) / (b-a) for x in array]
    (var) -> (predict)
end
# ---- Mean Normalization ----
@doc """
      Mean Normalization normalizes the data based on the mean.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.MeanNormalization(array)\n
      --------------------\n
      ==Functions==\n
      predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)
       """ ->
function MeanNormalization(array)
    avg = Lathe.stats.mean(array)
    a = minimum(array)
    b = maximum(array)
    predict(array) = [i = (i-avg) / (b-a) for i in array]
    (var) -> (predict)
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
      --------------------\n
      ==Functions==\n
      predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)
       """ ->
function StandardScalar(array)
    q = Lathe.stats.std(array)
    avg = Lathe.stats.mean(array)
    predict(array) = [i = (i-avg) / q for i in array]
    (var) -> (predict)
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
      df = DataFrame(:A => ['w','b','w'], :B => [5, 10, 15])\n
      scaled_feature = Lathe.preprocess.OneHotEncode(df,:A)\n
       """
function OneHotEncode(df,symb)
    copy = df
    for c in unique(copy[!,symb])
    copy[!,Symbol(c)] = copy[!,symb] .== c
    end
    predict(copy) = copy
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
