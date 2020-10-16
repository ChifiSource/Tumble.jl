#================
Preprocessing
     Module
================#
@doc """
      |====== Lathe.preprocess =====\n
      |____________/ Generalized Processing ___________\n
      |_____preprocess.TrainTestSplit\n
      |_____preprocess.SortSplit\n
      |_____preprocess.UniformSplit\n
      |____________/ Feature Scaling ___________\n
      |_____preprocess.Rescalar\n
      |_____preprocess.ArbitraryRescale\n
      |_____preprocess.MeanNormalization\n
      |_____preprocess.StandardScalar\n
      |____________/ Categorical Encoding ___________\n
      |_____preprocess.OneHotEncoder\n
      |_____preprocess.OrdinalEncoder\n
      |_____preprocess.FloatEncoder\n

       """ ->
module preprocess
using Random
using DataFrames
using Lathe.lstats
#==
Data Processing
==#
include("preprocess/data_processing.jl")
export TrainTestSplit, SortSplit, UniformSplit
#==
Encoding
==#
include("preprocess/encoding.jl")
export OneHotEncoder, OrdinalEncoder, FloatEncoder
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
