#==
Linear
    Least
     Square
==#
@doc """
      Least Squares is ideal for predicting continous features.
      Many models use Least Squares as a base to build off of.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      Type = :LIN\n
      model = Lathe.models.LeastSquare(x,y,Type)\n
      y_pred = Lathe.models.predict(model,xtrain)\n
      -------------------\n
      HYPER PARAMETERS\n
      Type <- Type determines which Linear Least Square algorithm to use,
      :LIN, :OLS, :WLS, and :GLS are the three options.\n
      - :LIN = Linear Least Square Regression\n
      - :OLS = Ordinary Least Squares\n
      - :WLS = Weighted Least Squares\n
      - :GLS = General Least Squares
      --------------------\n
      ==Functions==\n
      predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)
       """
function LeastSquare(x,y,Type)
    if length(x) != length(y)
        throw(ArgumentError("The array shape does not match!"))
    end
    if Type == :LIN
        xy = x .* y
        sxy = sum(xy)
        n = length(x)
        x2 = x .^ 2
        sx2 = sum(x2)
        sx = sum(x)
        sy = sum(y)
        # Calculate the slope:
        slope = ((n*sxy) - (sx * sy)) / ((n * sx2) - (sx)^2)
     # Calculate the y intercept
        b = (sy - (slope*sx)) / n
    end
    predict(xt) =
    if Type == :LIN
        (xt = [z = (slope * x) + b for x in xt])
    end
    (test)->(slope;b;predict)
end
#==
Ridge
    Regression
==#
@doc """
      Ridge Regression is another regressor ideal for predicting linear,
          continuous features.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.OneHotEncode(array)\n
       """
function RidgeRegression(x,y)

end
#==
Linear
    Regression
==#
@doc """
      Linear Regression is a well-known linear function used for predicting
      continuous features with a mostly linear or semi-linear slope.\n
      --------------------\n
      ==PARAMETERS==\n
     [y] <- Fill with your trainY values. Should be an array of shape (0,1) or (1,0)\n
     [x] <- Fill in with your trainX values. Should be an array of shape (0,1) or (1,0)\n
     --------------------\n
     ==Functions==\n
     predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)
       """
function LinearRegression(x,y)
    # a = ((∑y)(∑x^2)-(∑x)(∑xy)) / (n(∑x^2) - (∑x)^2)
    # b = (x(∑xy) - (∑x)(∑y)) / n(∑x^2) - (∑x)^2
    if length(x) != length(y)
        throw(ArgumentError("The array shape does not match!"))
    end
    # Get our Summations:
    Σx = sum(x)
    Σy = sum(y)
    # dot x and y
    xy = x .* y
    # ∑dot x and y
    Σxy = sum(xy)
    # dotsquare x
    x2 = x .^ 2
    # ∑ dotsquare x
    Σx2 = sum(x2)
    # n = sample size
    n = length(x)
    # Calculate a
    a = (((Σy) * (Σx2)) - ((Σx * (Σxy)))) / ((n * (Σx2))-(Σx^2))
    # Calculate b
    b = ((n*(Σxy)) - (Σx * Σy)) / ((n * (Σx2)) - (Σx ^ 2))
    # The part that is super struct:
    predict(xt) = (xt = [i = a + (b * i) for i in xt])
    (test)->(a;b;predict)
end
#==
Mean
    Baseline
==#
 # Model Type
 @doc """
       A mean baseline is great for getting a basic accuracy score in order
           to make a valid direction for your model.\n
         --------------------\n
         ==PARAMETERS==\n
        [y] <- Fill with your trainY values. Should be an array of shape (0,1) or (1,0)\n
        pipl = Pipeline([StandardScalar(),LinearRegression(trainX,trainy)])\n
        --------------------\n
        ==Functions==\n
        predict(xt) <- Returns a prediction from the model based on the xtrain value passed (xt)
                     """
function MeanBaseline(y)
    m = Lathe.stats.mean(m.y)
    predict(xt) =
    xt = [v = m for v in xt]
    (test)->(m;predict)
end
