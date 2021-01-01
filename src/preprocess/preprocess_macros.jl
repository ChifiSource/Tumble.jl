"""
      TrainTestSplits an Array\n
      --------------------\n
      x = [5,10,15,20]\n
      train, test = @tts x
       """
macro tts(x,at=.75)
    x = eval(x)
    TrainTestSplit(x,at)
end
"""
      Returns the normal distribution of an array.\n
      --------------------\n
      x = [5,10,15,20]\n
      norm = @norm x
       """
macro norm(x)
    x = eval(x)
    Normalizer(x).predict(x)
end
"""
      OneHotEncodes a dataframe\n
      Takes a symbol representing the column to one hot encode from and a DF.\n
      --------------------\n
      df = (:A => ["hello","world"], :B => ["Foo", "Bar"])\n
      encoded = @onehot df, :A
       """
macro onehot(df,symb)
    _onehotdf(df,symb)
end
