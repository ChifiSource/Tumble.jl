# ---- Rescalar (Standard Deviation) ---
@doc """
      Rescalar scales a feature based on the minimum and maximum of the array.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.Rescalar(array)\n
      --------------------\n
      ==Functions==\n
      predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)
       """ ->
function Rescalar(array)
    min = minimum(array)
    max = maximum(array)
    predict(array) = [i = (i-min) / (max - min) for i in array]
    (var) -> (predict)
end
# ---- Arbitrary Rescalar ----
@doc """
      Arbitrary Rescaling scales a feature based on the minimum and maximum
       of the array.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.Rescalar(array)\n
      --------------------\n
      ==Functions==\n
      predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)
       """ ->
function ArbitraryRescale(array)
    a = minimum(array)
    b = maximum(array)
    predict(array) = [x = a + ((i-a*i)*(b-a)) / (b-a) for x in array]
    (var) -> (predict)
end
# ---- Mean Normalization ----
@doc """
      Mean Normalization normalizes the data based on the mean.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.MeanNormalization(array)\n
      --------------------\n
      ==Functions==\n
      predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)
       """ ->
function MeanNormalization(array)
    avg = Lathe.stats.mean(array)
    a = minimum(array)
    b = maximum(array)
    predict(array) = [i = (i-avg) / (b-a) for i in array]
    (var) -> (predict)
end
# ---- Quartile Normalization ----
function QuartileNormalization(array)
    q1 = firstquar(array)
    q2 = thirdquar(array)

end
# ---- Z Normalization ----
@doc """
      Standard Scalar z-score normalizes a feature.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.StandardScalar(array)\n
      --------------------\n
      ==Functions==\n
      predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)
       """ ->
function StandardScalar(array)
    q = Lathe.stats.std(array)
    avg = Lathe.stats.mean(array)
    predict(array) = [i = (i-avg) / q for i in array]
    (var) -> (predict)
end
# ---- Unit L-Scale normalize ----
@doc """
      FUNCTION NOT YET WRITTEN\n
      Unit L Scaling uses eigen values to normalize the data.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.UnitLScale(array)\n
       """ ->
function UnitLScale(array)

end
