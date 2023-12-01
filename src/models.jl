
abstract type ModelType end

abstract type Categorical <: ModelType end

abstract type Classifier <: Categorical end

abstract type Continuous <: ModelType end

abstract type AbstractModel end

function fit! end

function predict end


mutable struct Model{T <: Any} <: AbstractModel
    data::Dict{String, Any}
    Model{T}() where {T <: ModelType} = new{T}(Dict{String, Any}())
end

function accuracy(model::Model{<:Continuous}, testx::Vector{<:Any}, testy::Vector{<:Any})
    yhat::Vector{<:Any} = predict(model, testx)
    (cor(yhat, testy) ^ 2)::Number
end

function accuracy(model::AbstractModel, testx::Vector{<:Any}, testy::Vector{<:Any})
    yhat::Vector{<:Any} = predict(model, testx)
    bitmask::Vector{Bool} = [begin 
        if x == y 
           true
        else 
           false
        end
    end for (x, y) in zip(yhat, testy)]
    (length(bitmask) / length(findall(x -> x, bitmask)))::Real
end

abstract type MajorityClass <: Categorical end

MajorityClass() = Model{MajorityClass}()

abstract type MeanBaseline <: Continuous end

MeanBaseline() = Model{MeanBaseline}()

function fit!(model::Model{MeanBaseline}, trainx::Vector{<:Any}, trainy::Vector{<:Any})
    push!(model.data, "mu" => Float64(sum(trainy) / length(trainy)))
end

predict(model::Model{MeanBaseline}, testx) = begin
    if ~("mu" in keys(model.data))
        throw("this model has not been fitted to data yet. Use `fit!` first.")
    end
    mean::Real = model.data["mu"]
    [mean for x in testx]::Vector{<:Any}
end
