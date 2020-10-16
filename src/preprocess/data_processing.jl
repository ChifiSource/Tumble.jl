using Random
using DataFrames
# Train-Test-Split-----
function _dfTrainTestSplit(df,at = 0.75)
    sample = randsubseq(1:size(df,1), at)
    trainingset = df[sample, :]
    notsample = [i for i in 1:size(df,1) if isempty(searchsorted(sample, i))]
    testset = df[notsample, :]
    return(trainingset,testset)
end
function _ArraySplit(data, at = 0.75)
    n = length(data)
    idx = Random.shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
    return(test_idx,train_idx)
end
@doc """
      Train test split is a method used to divide data into smaller fragments
       for testing, training, and validation. The method can take a dataframe
           or an array.\n
      --------------------\n
      df = DataFrame(:A => [5,10,15], :B => [5,10,15])\n
      train, test = TrainTestSplit(df)
      --------------------\n
      PARAMETERS:\n
      data:: Iterable dictionary, dataframe, or Array.\n
      at:: Percentage value used to determine a point to split the data.\n
       """
TrainTestSplit(data::Array, at::Float64=.75) = _ArraySplit(data,at)
TrainTestSplit(data::DataFrame, at::Float64=.75) = _dfTrainTestSplit(data,at)
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
      Uniform Split does the exact same thing as TrainTestSplit, except
      observations are returned unsorted and unshuffled.\n
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
