#================
Predictive
    Learning
        Models
================#
@doc """
| Model Type | Model Name |
|:---------- | ---------- |
| Baseline    | MeanBaseline |
| Baseline    | ClassBaseline |
| Continuous    | LinearRegression |
| Continuous    | LinearLeastSquare  |
| Categorical    | RandomForestClassifier  |
| Categorical    | DecisionTree  |
| Tools    | Pipeline  |
| Tools    | PowerLog  |
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
abstract type BaselineModel <: Model end
# Unsupervised
abstract type UnsupervisedModel <: Model end
#==
Continuous Models
==#
include("models/regressors.jl")
export LinearLeastSquare
export LinearRegression
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
