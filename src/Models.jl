#================
Predictive
    Learning
        Models
================#
@doc """
| Model Type | Model Name |\n
|:---------- | ---------- |\n
| Baseline    | MeanBaseline |\n
| Baseline    | ClassBaseline |\n
| Continuous    | LinearRegression |\n
| Continuous    | LinearLeastSquare  |\n
| Categorical    | RandomForestClassifier  |\n
| Categorical    | DecisionTree  |\n
| Tools    | Pipeline  |\n
| Tools    | PowerLog  |
       """
module models
# [deps]
using Random
using DataFrames
using Lathe.stats
# [deps]
# Type heirarchy
abstract type Model <: LatheObject end
abstract type SupervisedModel <: Model end
abstract type ContinuousModel <: SupervisedModel end
abstract type Regressor <: ContinuousModel end
abstract type LinearModel <: Regressor end
abstract type CategoricalModel <: Model end
abstract type BaselineModel <: Model end
abstract type Tool <: Model end
abstract type UnsupervisedModel <: Model end
abstract type Classifier <: CategoricalModel end
#==
Continuous Models
==#
include("models/regressors.jl")
export LinearLeastSquare
export LinearRegression
export LassoRegression
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
Tools
==#
include("models/toolbox.jl")
export Pipeline, PowerLog, MeanBaseline, majClassBaseline
include("models/throws.jl")
#----------------------------------------------
end
