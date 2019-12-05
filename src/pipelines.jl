# <--- Lathe Pipelines --->
# [deps]
# <--- Lathe Pipelines --->
module pipelines
using Lathe
mutable struct Pipeline
    steps
    model
end
function pippredict(pipe,xt)
    println("---DEPRECATION WARNING---")
    println("Lathe.pipelines is set to be deprecated in Lathe 0.1.0")
    println("! Use Pipeline([steps],model) from Lathe.models")
    println("julia> using Lathe.models: Pipeline, predict")
    println("pipe = Pipeline([StandardScalar],LinearRegression(StandardScalar(trainX),trainy))
    println("predict(pipe,Xtrain)")
    for step in pipe.steps
        xt = step(xt)
    end
    ypr = Lathe.models.predict(pipe.model,xt)

    return(ypr)
end
#------------------
end
