abstract type Weights end

struct NormalWeights <: Weights
    weights
end
struct BinomialWeight <: Weights
    weights
end
struct NoWeights <: Weights
    weights
end
function NormalWeight(df::DataFrame, y::Array{Any})
    features = eachcol(df)
    n_features = size(df)[1]
    categories = Set(y)
    y = OrdinalEncoder(y).predict(y)
    dist = NormalDist(y)
end
function BinomialWeight(df::DataFrame)

end
