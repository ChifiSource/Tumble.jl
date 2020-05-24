# <---- One Hot Encoder ---->

@doc """
      One hot encoder replaces a single feature with sub arrays containing
      boolean values (1 or 0) for each individual category.\n
      --------------------\n
      df = DataFrame(:A => ['w','b','w'], :B => [5, 10, 15])\n
      scaled_feature = Lathe.preprocess.OneHotEncode(df,:A)\n
       """
function OneHotEncoder()
    predict(copy) = _onehot(copy)
    ()->(predict)
end
function _onehotdf(df,symb)
    copy = df
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
# <---- Invert Encoder ---->
#==
@doc """
      FUNCTION NOT YET WRITTEN\n
      Invert Encoder (Not written.)\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.OneHotEncode(array)\n
       """ ->
       ==#
function InvertEncode(array)

end
