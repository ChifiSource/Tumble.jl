# <---- One Hot Encoder ---->

@doc """
      One hot encoder replaces a single feature with sub arrays containing
      boolean values (1 or 0) for each individual category.\n
      --------------------\n
      df = DataFrame(:A => ['w','b','w'], :B => [5, 10, 15])\n
      scaled_feature = Lathe.preprocess.OneHotEncode(df,:A)\n
       """
function OneHotEncode(df,symb)
    copy = df
    for c in unique(copy[!,symb])
    copy[!,Symbol(c)] = copy[!,symb] .== c
    end
    predict(copy) = copy
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
