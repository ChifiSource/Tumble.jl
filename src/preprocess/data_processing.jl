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
    return(train_idx, test_idx)
end
"""
    ## TrainTestSplit
    ### Description
      Splits an array or dataframe into two smaller groups based on the
      percentage provided in the at parameter.\n
      --------------------\n
    ### Input
      TrainTestSplit(x, .75)\n
      --------------------\n
      #### Positional Arguments
      Array{Any}, DataFrame - data:: The data to split.\n
      Float64 - at:: A percentage that determines where the data is split.
      --------------------\n
     ### Output
     train:: The larger half of the split set.\n
     test:: The smaller half of the split set.
       """
TrainTestSplit(data::Array, at::Float64=.75) = _ArraySplit(data,at)
TrainTestSplit(data::DataFrame, at::Float64=.75) = _dfTrainTestSplit(data,at)
mutable struct Splitter{P} <: Manager
    at::Float64
    predict::P
    function Splitter(at::Float64)
        predict(x) = TrainTestSplit(x, at)
        new{typeof(predict)}(at, predict)
    end
end
# Sort-Split -------------
"""
    ## Sort Split
    ### Description
      Sorts an array, and then splits said array.\n
      --------------------\n
    ### Input
      SortSplit(x, .75, false)\n
      --------------------\n
      #### Positional Arguments
      Array{Any} - data:: The data to split.\n
      Float64 - at:: A percentage that determines where the data is split.\n
      Bool - rev:: Determines whether the order of the sort should be reversed.\n
      --------------------\n
     ### Output
     train:: The larger half of the split set.\n
     test:: The smaller half of the split set.
       """
function SortSplit(data, at = 0.25, rev=false)
  n = length(data)
  sort!(data, rev=rev)  # Sort in-place
  train_idx = view(data, 1:floor(Int, at*n))
  test_idx = view(data, (floor(Int, at*n)+1):n)
  return(train_idx, test_idx)
end
# Unshuffled Split ----
"""
    ## Uniform Split
    ### Description
      Uniform Split will split an array without shuffling the data first.\n
      --------------------\n
    ### Input
      UniformSplit(x, .75)\n
      --------------------\n
      #### Positional Arguments
      Array{Any} - data:: The data to split.\n
      Float64 - at:: A percentage that determines where the data is split.\n
      --------------------\n
     ### Output
     train:: The larger half of the split set.\n
     test:: The smaller half of the split set.
       """
function UniformSplit(data, at = 0.7)
    n = length(data)
    idx = data
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
    return(train_idx, test_idx)
end
