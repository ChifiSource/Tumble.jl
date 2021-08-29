mutable struct QuantileNormalizer <: Normalizer
    q1::Real
    q3::Real
    predict::Function
    function QuantileNormalizer(x::Array, lower::Float64 = .25,
         upper::Float64 = .75)
        q1 = quantile(x, lower)
        q3 = quantile(x, upper)
        predict(x::Array) = qn_pred(x, q1, q3)
        new(q1, q3, predict)
    end
    function qn_pred(x, q1, q3)
        [if i > q3 x[current] = q3 elseif i < q1 x[current] = q1
            else x[current] = i end for (current, i) in enumerate(x)]
    end
end
mutable struct ZNormalizer <: Normalizer
    scaler::LatheObject
    predict::Function
    function ZNormalizer(x::Array)
        scaler = StandardScaler(x)
        predict(x::Array) = zn_pred(scaler, x)
        new(scaler, predict)
    end
    function zn_pred(scaler::LatheObject, input)
        [if i > 1.8 x[current] = 1.8 elseif i < -1.8 x[current] = -1.8
            else x[current] = i end for (current, i) in enumerate(x)]
    end
end
