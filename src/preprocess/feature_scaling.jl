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
mutable struct MeanScaler{P} <: Scaler
    predict::P
    avg::Float64
    a::Float64
    b::Float64
    function MeanScaler(array::Array)
        avg = mean(array)
        a = minimum(array)
        b = maximum(array)
        predict(array::Array) = [i = (i-avg) / (b-a) for i in array]
        predict(df::DataFrame, symb::Symbol) = [i = (i-avg) / (b-a) for i in df[!, symb]]
        P = typeof(predict)
        return new{P}(predict, avg, a, b)
    end
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
mutable struct StandardScaler{dist, predict} <: Scaler
    dist::Distribution
    predict::predict
    function StandardScaler(x::Array)
        dist = NormalDist(x)
        predict(xt::Array) = dist.apply(xt)
        predict(df::DataFrame, symb::Symbol) = dist.apply(df[!, symb])
        D, P =  typeof(dist), typeof(predict)
        return new{D, P}(dist, predict)
    end
end

function QuantileTransformer(array)
    norm = NormalDist(array)
    normalized = norm.apply(array)
    dist = UniformDist(normalized)
    predict(xt) = dist.cummulative(xt)
    (var) -> (dist;predict;norm)
end
