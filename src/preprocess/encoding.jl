# <---- One Hot Encoder ---->

@doc """
      One hot encoder replaces a single feature with sub arrays containing
      boolean values (1 or 0) for each individual category.\n
      --------------------\n
      ==PARAMETERS==\n
      (This function has no parameters)\n
      --------------------\n
      ==FUNCTIONS==\n
      predict(df, symb) <- returns a copy of the dataframe with
       """
function OneHotEncoder()
    predict(df, symb) = _onehot(df,symb)
    ()->(predict)
end

function _onehot(df,symb)
    copy = df
    copy = [copy[c] = copy[c] .== c for c in unique(copy)]
    return(copy)
end
function OrdinalEncoder(array)
    uni = Set(array)
    lookup = Dict()
    [push!(lookup, (value => i)) for (i, value) in enumerate(uni)]
    predict() = [row = lookup[row] for row in array]
    ()->(predict)
end
function FloatEncoder(array)
    encoded_array = []

    for dim in array
        newnumber = 0
        for char in dim
            newnumber += Float64(char)
        end
        append!(encoded_array, newnumber)
    end
    predict() = encoded_array
    ()->(predict)
end
