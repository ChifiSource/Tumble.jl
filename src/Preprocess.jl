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
# [deps]
using Random
using DataFrames
using Lathe.stats
# [deps]
abstract type LatheObject end
abstract type Preprocessor <: LatheObject end
abstract type Encoder <: Preprocessor end
abstract type Scaler <: Preprocessor end
abstract type Transformer <: Preprocessor end
abstract type Manager <: Preprocessor end
abstract type Booster <: Preprocessor end
#==
Data Processing
==#
include("preprocess/data_processing.jl")
#==
Encoding
==#
include("preprocess/encoding.jl")
#==
Feature Scaling
==#
include("preprocess/feature_scaling.jl")
#==
Macros
==#
include("preprocess/preprocess_macros.jl")
#==
Decompisition
==#
include("preprocess/decomp.jl")
#-----------------------------
end
