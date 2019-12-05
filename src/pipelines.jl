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
    for step in pipe.steps
        xt = step(xt)
    end
    ypr = Lathe.models.predict(pipe.model,xt)
    return(ypr)
end
function serialize(pip,uri)

end
function deserialize(pip,uri)

end
#------------------
end
