import Random: Sampler
using Random
function direct_sample!(rng::AbstractRNG, a::UnitRange, x::AbstractArray)
    s = Sampler(rng, 1:length(a))
    b = a[1] - 1
    if b == 0
        for i = 1:length(x)
            @inbounds x[i] = rand(rng, s)
        end
    else
        for i = 1:length(x)
            @inbounds x[i] = b + rand(rng, s)
        end
    end
    return x
end
function sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray;
                 replace::Bool=true, ordered::Bool=false)
    n = length(a)
    k = length(x)
    k == 0 && return x

    if replace  # with replacement
        if ordered
            sort!(direct_sample!(rng, a, x))
        else
            direct_sample!(rng, a, x)
        end

    else  # without replacement
        k <= n || error("Cannot draw more samples without replacement.")

        if ordered
            if n > 10 * k * k
                seqsample_c!(rng, a, x)
            else
                seqsample_a!(rng, a, x)
            end
        else
            if k == 1
                @inbounds x[1] = sample(rng, a)
            elseif k == 2
                @inbounds (x[1], x[2]) = samplepair(rng, a)
            elseif n < k * 24
                fisher_yates_sample!(rng, a, x)
            else
                self_avoid_sample!(rng, a, x)
            end
        end
    end
    return x
end
direct_sample!(a::UnitRange, x::AbstractArray) = direct_sample!(Random.GLOBAL_RNG, a, x)
function sample(rng::AbstractRNG, a::AbstractArray{T}, n::Integer;
                replace::Bool=true, ordered::Bool=false) where T
    sample!(rng, a, Vector{T}(undef, n); replace=replace, ordered=ordered)
end
sample(a::AbstractArray, n::Integer; replace::Bool=true, ordered::Bool=false) =
    sample(Random.GLOBAL_RNG, a, n; replace=replace, ordered=ordered)

    sample(rng::AbstractRNG, a) = a[rand(rng, 1:length(a))]
    sample(a::AbstractArray) = sample(Random.GLOBAL_RNG, a)
