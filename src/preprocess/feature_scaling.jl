# ---- Rescalar (Standard Deviation) ---
"""
    ## Rescalar
    ### Description
      Rescales an array.\n
      --------------------\n
    ### Input
      Rescaler(x)\n
      --------------------\n
      #### Positional Arguments
      Array{Any} - x:: Array for which the original scaler should be based
      off of.\n
      --------------------\n
     ### Output
     scalar :: A Lathe Preprocesser object.
     ---------------------\n
     ### Functions
     Preprocesser.predict(xt) :: Applies the scaler to xt.\n
     ---------------------\n
     ### Data
     min :: The minimum value in the array.\n
     max :: The maximum value in the array.
       """
function Rescaler(array)
    min = minimum(array)
    max = maximum(array)
    predict(array) = [i = (i-min) / (max - min) for i in array]
    (var) -> (predict;min;max)
end
# ---- Arbitrary Rescalar ----
"""
    ## Arbitrary Rescaler
    ### Description
      Arbitrarily rescales an array.\n
      --------------------\n
    ### Input
      ArbitraryRescaler(x)\n
      --------------------\n
      #### Positional Arguments
      Array{Any} - x:: Array for which the original scaler should be based
      off of.\n
      --------------------\n
     ### Output
     scalar :: A Lathe Preprocesser object.
     ---------------------\n
     ### Functions
     Preprocesser.predict(xt) :: Applies the scaler to xt.\n
     ---------------------\n
     ### Data
     a :: The minimum value in the array.\n
     b :: The maximum value in the array.
       """
function ArbitraryRescaler(array)
    a = minimum(array)
    b = maximum(array)
    predict(array) = [x = a + ((i-a*i)*(b-a)) / (b-a) for x in array]
    (var) -> (predict;a;b)
end
# ---- Mean Normalization ----
"""
    ## Mean Normalizer
    ### Description
      Normalizes an array using the mean of the data.\n
      --------------------\n
    ### Input
      ArbitraryRescaler(x)\n
      --------------------\n
      #### Positional Arguments
      Array{Any} - x:: Array for which the original scaler should be based
      off of.\n
      --------------------\n
     ### Output
     scalar :: A Lathe Preprocesser object.
     ---------------------\n
     ### Functions
     Preprocesser.predict(xt) :: Applies the scaler to xt.\n
     ---------------------\n
     ### Data
     a :: The minimum value in the array.\n
     b :: The maximum value in the array.\n
     avg :: The mean of the array.
       """
function MeanNormalizer(array)
    avg = mean(array)
    a = minimum(array)
    b = maximum(array)
    predict(array) = [i = (i-avg) / (b-a) for i in array]
    (var) -> (predict;avg;a;b)
end
# ---- Z Normalization ----
"""
    ## Standard Scaler
    ### Description
      Normalizes an array using the z (Normal) distribution.\n
      --------------------\n
    ### Input
      StandardScaler(x)\n
      --------------------\n
      #### Positional Arguments
      Array{Any} - x:: Array for which the original scaler should be based
      off of.\n
      --------------------\n
     ### Output
     scalar :: A Lathe Preprocesser object.
     ---------------------\n
     ### Functions
     Preprocesser.predict(xt) :: Applies the scaler to xt.\n
     ---------------------\n
     ### Data
     dist  :: Returns the normal distribution object for which this scaler uses.
       """
function StandardScaler(array)
    dist = NormalDist(array)
    predict(xt) = dist.apply(xt)
    (var) -> (predict;dist)
end

function QuantileTransformer(array)
    norm = NormalDist(array)
    normalized = norm.apply(array)
    dist = UniformDist(normalized)
    predict(xt) = dist.cummulative(xt)
    (var) -> (dist;predict;norm)
end
