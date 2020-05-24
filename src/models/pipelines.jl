@doc """
      Pipelines can contain a predictable Lathe model with preprocessing that
      occurs automatically. This is done by putting X array processing methods
      into the iterable steps, and then putting your Lathe model in.\n
      --------------------\n
      ==PARAMETERS==\n
      [steps] <- An iterable list of methods to call for X modification. These mutations should
      have ALREADY BEEN MADE TO THE TRAIN X.\n
      pipl = Pipeline([StandardScalar(),LinearRegression(trainX,trainy)])\n
      --------------------\n
      ==Functions==\n
      predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)
      """
function Pipeline(steps)
    predict(xt) = [object.predict(xt) for object in steps]
    (var)->(steps;model;predict)
end
