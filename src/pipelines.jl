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
    for step in pipe.steps
        u = step(m.x)
        append!(fx,u)
    end
    m.x = fx
    pr = Lathe.models.predict(model,xt)
    return(pr)
end
function serialize(pip,uri)

end
function deserialize(pip,uri)

end
#------------------
end
