# <---- One Hot Encoder ---->

"""
    ## OneHotEncoder
    ### Description
      One Hot Encodes a dataframe column into a dataframe.\n
      --------------------\n
    ### Input
      OneHotEncoder()\n
      --------------------\n
     ### Output
     encoder :: A Lathe Preprocesser object.
     ---------------------\n
     ### Functions
     Preprocesser.predict(df, symb) :: Applies the encoder to the dataframe key
      corresponding with symb on DF, then returns a dataframe with encoded
      results.
       """
function OneHotEncoder()
    predict(df, symb) = _onehot(df,symb)
    ()->(predict)
end

function _onehot(df,symb)
    copy = copy(df)
    copy = [copy[c] = copy[c] .== c for c in unique(copy)]
    return(copy)
end
"""
    ## Ordinal Encoder
    ### Description
     Ordinally Encodes an array.\n
      --------------------\n
    ### Input
      OrdinalEncoder(x)\n
      --------------------\n
      #### Positional Arguments
      Array{Any} - x:: Array for which the original scaler should be based
      off of.\n
      --------------------\n
     ### Output
     encoder :: A Lathe Preprocesser object.
     ---------------------\n
     ### Functions
     Preprocesser.predict(xt) :: Returns an ordinally encoded xt.\n
       """
function OrdinalEncoder(array)
    uni = Set(array)
    lookup = Dict()
    [push!(lookup, (value => i)) for (i, value) in enumerate(uni)]
    predict(arr) = [row = lookup[row] for row in arr]
    ()->(predict;lookup)
end
"""
    ## Float Encoder
    ### Description
     Float/Label Encodes an array.\n
      --------------------\n
    ### Input
      OneHotEncoder()\n
      --------------------\n
     ### Output
     encoder :: A Lathe Preprocesser object.
     ---------------------\n
     ### Functions
     Preprocesser.predict(xt) :: Returns an ordinally encoded xt.\n
       """
function FloatEncoder()
    predict(xt) = _floatencode(xt)
    ()->(predict)
end
function _floatencode(array)
    encoded_array = []
    for dim in array
        newnumber = 0
        for char in dim
            newnumber += Float64(char)
        end
        append!(encoded_array, newnumber)
    end
    return(encoded_array)
end
