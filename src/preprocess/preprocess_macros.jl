macro tts(x,y,at=.75)
    x = eval(x)
    y = eval(y)
    TrainTestSplit(x,y,at)
end
macro onehot(df,symb)
    OneHotEncode(df,symb)
end
macro norm(x)
    x = eval(x)
    Normalizer(x).predict(x)
end
