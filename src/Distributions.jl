include("Stats.jl")
include("Preprocess.jl")
normal(var) = preprocess.StandardScalar(var)
export normal
