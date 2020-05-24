macro tts(x,y,at=.75)
    x = eval(x)
    y = eval(y)
    TrainTestSplit(x,y,at)
end
macro norm(x)
    x = eval(x)
    Normalizer(x).predict(x)
end
macro onehot(df,symb)
    _onehotdf(df,symb)
end
