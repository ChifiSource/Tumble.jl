macro tts(x,at=.75)
    x = eval(x)
    TrainTestSplit(x,at)
end
macro norm(x)
    x = eval(x)
    Normalizer(x).predict(x)
end
macro onehot(df,symb)
    _onehotdf(df,symb)
end
