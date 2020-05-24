macro predict(m,trainx,trainy,xt)
    expression = string(m,"(", trainx, ",", trainy, ")")
    exp = Meta.parse(expression)
    model = eval(exp)
    xt = eval(xt)
    pred = model.predict(xt)
end
