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
using DataFrames
using Lathe.stats
#==
Data Loading
==#
include("preprocess/data_loading.jl")
#==
Data Processing
==#
include("preprocess/data_processing.jl")
export TrainTestSplit, SortSplit, UniformSplit
#==
Encoding
==#
include("preprocess/encoding.jl")
export OneHotEncode, InvertEncode
#==
Feature Scaling
==#
include("preprocess/feature_scaling.jl")
export Rescalar, ArbitraryRescale, MeanNormalization, QuartileNormalization
export Standardize, UnitLScale
#==
Macros
==#
include("preprocess/preprocess_macros.jl")
export @tts, @norm, @onehot
#-----------------------------
end
