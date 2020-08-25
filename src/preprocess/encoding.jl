# <---- One Hot Encoder ---->

@doc """
      One hot encoder replaces a single feature with sub arrays containing
      boolean values (1 or 0) for each individual category.\n
      --------------------\n
      df = DataFrame(:A => ['w','b','w'], :B => [5, 10, 15])\n
      scaled_feature = Lathe.preprocess.OneHotEncode(df,:A)\n
       """
function OneHotEncoder(df,symb)
    copy = df

    predict() = _onehotdf(df,symb)
    ()->(predict;df;symb)
end
function _onehotdf(df,symb)
    for c in unique(copy[!,symb])
        copy[!,Symbol(c)] = copy[!,symb] .== c
    end
    return(copy)
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
