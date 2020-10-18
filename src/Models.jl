#================
Predictive
    Learning
        Models
================#
@doc """
      |====== Lathe.models =====\n
      |____________/ Accessories ___________\n
      |_____models.Pipeline([steps])\n
      |_____models.Powerlog(p1, p2)\n
      |____________/ Continuous models ___________\n
      |_____models.meanBaseline(y)\n
      |_____models.SimpleLinearRegression(x,y)\n
      |_____models.LeastSquare(x,y,Type)\n
      |_____models.PowerLog(prob1,prob2)\n
      |____________/ Categorical Models ___________\n
      |_____models.LogisticRegression(x,y)\n
      |_____models.majBaseline(y)\n
      |____________/ Unsupervised Learning Models ___________\n
      |_____models.Kmeans(k)\n

       """
module models
# [deps]
using Random
using DataFrames
using Lathe.stats
# [deps]
#==
Continuous Models
==#
# Least Square
include("models/lsq.jl")
# Linear Regression
include("models/linreg.jl")
#==
Categorical Models
==#
# Decision Tree Classifier/ RF
include("models/treeclass.jl")
# Logistic Regression
include("models/logistic.jl")
#==
Unsupervised Models
==#
include("models/kmeans.jl")
#==
Tools
==#
include("models/toolbox.jl")
#==
Macros
==#
include("models/model_macros.jl")
#==
Neural Networks
==#
include("models/neural.jl")
#----------------------------------------------
end
