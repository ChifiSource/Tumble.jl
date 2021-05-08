using CUDA
function checkdims(x::AbstractArray, y::AbstractArray)
    @assert length(x) == length(Array(y)) DimensionMismatch("Shape does not match!")
end
function cudacheck(arrays, cuda::Bool)
    if cuda == true
        new = [x = CuArray(x) for x in arrays]
        return(new)
    else
        return(arrays)
    end
end
