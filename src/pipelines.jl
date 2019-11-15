# <--- Lathe Pipelines --->
# [deps]
# <--- Lathe Pipelines --->
module pipelines
using Lathe
mutable struct Pipeline
    steps
    model
end
function predict(pipe,xt)
    fx = []
    m = pipe.model
    for i in pipe.steps
        u = i(m.x)
        append!(fx,u)
    end
    model.x = fx
    pr = Lathe.models.predict(model,xt)
    return(pr)
end
#------------------
end
