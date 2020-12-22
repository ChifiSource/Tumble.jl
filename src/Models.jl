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
using Lathe.lstats
# [deps]
abstract type Model end
# Supervised
abstract type SupervisedModel <: Model end
abstract type ContinuousModel <: SupervisedModel end
abstract type LinearModel <: ContinuousModel end
abstract type CategoricalModel <: Model end
# Unsupervised
abstract type UnsupervisedModel <: Model end
#==
Continuous Models
==#
# Least Square
include("models/lsq.jl")
export LeastSquare
# Linear Regression
include("models/linreg.jl")
export SimpleLinearRegression
#==
Categorical Models
==#
# Decision Tree Classifier/ RF
include("models/treeclass.jl")
export DecisionTreeClassifier, RandomForestClassifier
# Logistic Regression
include("models/logistic.jl")
export LogisticRegression
#==
Unsupervised Models
==#
include("models/kmeans.jl")
#==
Tools
==#
include("models/toolbox.jl")
export Pipeline, PowerLog, MeanBaseline, majClassBaseline
#==
Neural Networks
==#
include("models/neural.jl")
export Network
#----------------------------------------------
end
