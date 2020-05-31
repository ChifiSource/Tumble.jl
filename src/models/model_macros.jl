""" The predict macro will take a model name (as a string), a trainX and trainy
array, and an x train array <-- in that order, and then return a prediction from
said model. This might be ideal to get predictions out fast, but it will make
hyper-parameter optimization impossible.\n
-------------\n
m = "LinearRegression"
trainx = [5,10,15]
trainy = [5,10,15]
xtrain = [5,10,15]
@predict m trainx trainy xtrain
"""
macro predict(m,trainx,trainy,xt)
    expression = string(m,"(", trainx, ",", trainy, ")")
    exp = Meta.parse(expression)
    model = eval(exp)
    xt = eval(xt)
    model.predict(xt)
end
