#================
Predictive
    Learning
        Models
================#
@doc """
      |====== Lathe.models =====\n
      |____________/ Accessories ___________\n
      |_____models.Pipeline([steps])\n
      |____________/ Continuous models ___________\n
      |_____models.meanBaseline(y)\n
      |_____models.LinearRegression(x,y)\n
      |_____models.LeastSquare(x,y,Type)\n
      |_____models.PowerLog(prob1,prob2)\n
      |____________/ Categorical Models ___________\n
      |_____models.LogisticRegression(x,y)\n
      |_____models.majBaseline(y)\n
       """
module models
# [deps]
using Lathe
using Random
using DataFrames
# [deps]
#==
Linear Models
==#
include("models/linear_models.jl")
export LinearRegression, LeastSquare, RidgeRegression, MeanBaseline
#==
Categorical Models
==#
include("models/cat_models.jl")
export majClassBaseline, LogisticRegression
#==
Pipelines
==#
include("models/pipelines.jl")
export Pipeline
#==
Tools
==#
include("models/toolbox.jl")
export PowerLog
#==
Macros
==#
include("models/model_macros.jl")
#==
Neural Networks
==#
include("models/neural.jl")
export Network
#==
Nonlinear Models
==#
include("models/nonlinear_models.jl")
#----------------------------------------------
end
