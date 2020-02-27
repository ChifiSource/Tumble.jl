#================
Predictive
    Learning
        Models
================#
@doc """
      |====== Lathe.models =====\n
      |____________/ Accessories ___________\n
      |_____models.predict(m,xt)\n
      |_____models.Pipeline([steps],model)\n
      |____________/ Continuous models ___________\n
      |_____models.meanBaseline(y)\n
      |_____models.RegressionTree(x,y,n_divisions)\n
      |_____models.FourSquare(x,y)\n
      |_____models.IsotonicRegression(x,y)\n
      |_____models.MultipleLinearRegression([x],y)\n
      |_____models.RidgeRegression(x,y)\n
      |_____models.LinearRegression(x,y)\n
      |_____models.LeastSquare(x,y,Type)\n
      |____________/ Categorical Models ___________\n
      |_____models.LogisticRegression(x,y)\n
      |_____models.majBaseline\n
       """ ->
module models
#==
Base
    Models
        Functions
==#
using Lathe
using Random
#===========
Accessories
===========#
@doc """
      Pipelines can contain a predictable Lathe model with preprocessing that
      occurs automatically. This is done by putting X array processing methods
      into the iterable steps, and then putting your Lathe model in.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      model = Lathe.models.meanBaseline(y)\n
      StandardScalar = Lathe.preprocess.StandardScalar\n
      MeanNormalization = Lathe.preprocess.MeanNormalization\n
      steps = [StandardScalar,MeanNormalization]\n
      pipeline = Lathe.models.Pipeline(steps,model)\n
      y_pred = Lathe.models.predict(pipeline,xtrain)\n
      --------------------\n
      HYPER PARAMETERS\n
      steps:: Iterable list (important, use []) of processing methods to be
      performed on the xtrain set. Note that it will not be applied to the
      train set, so preprocessing for the train set should be done before
      model construction.\n
      model:: Takes any Lathe model, uses Lathe.models.predict,\n
      method assersion is still do-able with the dispatch, meaning any model\n
      designed to work with Lathe.models (and Lathe.models.predict) will work\n
      inside of a Lathe pipeline."""
mutable struct Pipeline
    steps
    model
end
function pred_pipeline(m,x)
    for step in m.steps
        x = step(x)
    end
    x = [x = step(x) for step in m.steps]
    ypr = model.predict(x)
    return(ypr)
end
#==============
========================================================
=======================================================================
            CONTINUOS MODELS               CONTINUOS MODELS
            CONTINUOS MODELS               CONTINUOS MODELS
======================================================================
======================================================================#
#==
Mean
    Baseline
==#
 # Model Type
 @doc """
       A mean baseline is great for getting a basic accuracy score in order
           to make a valid direction for your model.\n
       --------------------\n
       x = [7,6,5,6,5]\n
       y  = [3.4.5.6.3]\n
       xtrain = [7,5,4,5,3,5,7,8]\n
       model = Lathe.models.meanBaseline(y)
       y_pred = Lathe.models.predict(model,xtrain)\n
        """ ->
mutable struct meanBaseline
    y
end
#----  Callback
function pred_meanbaseline(m,xt)
    e = []
    m = Lathe.stats.mean(m.y)
    print("-Lathe.models Mean Baseline-")
    print("mean: ",m)
    for i in xt
        append!(e,m)
    end
    return(e)
end
#==
Regression
    Tree
==#
# Model Type
@doc """
      Pipelines can contain a predictable Lathe model with preprocessing that
      occurs automatically.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      n_divisions = 4\n
      model = Lathe.models.RegressionTree(x,y,n_divisions)\n
      --------------------\n
      HYPER PARAMETERS\n
      n_divisions:: n_divisions determines the number of divisions that the
      regression tree should take."""
mutable struct RegressionTree
    x
    y
    n_divisions
end
#----  Callback
function pred_regressiontree(m,xt)

end
#==
Four
    Square
==#
@doc """
      A FourSquare splits data into four linear least squares, and then
      predicts variables depending on their location in the data (in
      quartile range.) With the corresponding model for said quartile.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      model = Lathe.models.FourSquare(x,y)\n"""
      #==
      Four
          Square
      ==#
      # Model Type
      mutable struct FourSquare
          x
          y
      end
      #----  Callback
      function pred_foursquare(m,xt)
          # x = q1(r(floor:q1)) |x2 = q2(r(q1:μ)) |x3 = q3(r(q2:q3)) |x4 q4(r(q3:cieling))
          # y' = q1(x * (a / x)) | μ(x * (a / x2)) | q3(x * (a / x3) | q4(x * (a / x4))
              x = m.x
              y = m.y
              # Go ahead and throw an error for the wrong input shape:
              xlength = length(x)
              ylength = length(y)
              if xlength != ylength
                  throw(ArgumentError("The array shape does not match!"))
              end
              # Our empty Y prediction list==
              e = []
              # Quad Splitting the data ---->
              # Split the Y
              y2,range1 = Lathe.preprocess.SortSplit(y)
              y3,range2 = Lathe.preprocess.SortSplit(y2)
              y4,range3 = Lathe.preprocess.SortSplit(y3)
              y5,range4 = Lathe.preprocess.SortSplit(y4)
              yrange5 = y5
              # Split the x train
              x1,xrange1 = Lathe.preprocess.SortSplit(x)
              x2,xrange2 = Lathe.preprocess.SortSplit(x1)
              x3,xrange3 = Lathe.preprocess.SortSplit(x2)
              x4,xrange4 = Lathe.preprocess.SortSplit(x3)
              xrange5 = y5
              # Fitting the 4 linear regression models ---->
              regone = LeastSquare(xrange1,range1,:LIN)
              regtwo = LeastSquare(xrange2,range2,:LIN)
              regthree = LeastSquare(xrange3,range3,:LIN)
              regfour = LeastSquare(xrange4,range4,:LIN)
              regfive = LeastSquare(xrange5,yrange5,:LIN)
              # Split the train Data
              xt1,xtrange1 = Lathe.preprocess.SortSplit(xt)
              xt2,xtrange2 = Lathe.preprocess.SortSplit(xt1)
              xt3,xtrange3 = Lathe.preprocess.SortSplit(xt2)
              xt4,xtrange4 = Lathe.preprocess.SortSplit(xt3)
              xtrange5 = xt4
              # Get min-max
              xtrange1min = minimum(xtrange1)
              xtrange1max = maximum(xtrange1)
              xtrange2min = minimum(xtrange2)
              xtrange2max = maximum(xtrange2)
              xtrange3min = minimum(xtrange3)
              xtrange3max = maximum(xtrange3)
              xtrange4min = minimum(xtrange4)
              xtrange4max = maximum(xtrange4)
              xtrange5min = minimum(xtrange5)
              # Ranges for ifs
              condrange1 = (xtrange1min:xtrange1max)
              condrange2 = (xtrange2min:xtrange2max)
              condrange3 = (xtrange3min:xtrange3max)
              condrange4 = (xtrange4min:xtrange4max)
              # This for loop is where the dimension's are actually used:
              for i in xt
                  if i in condrange1
                      ypred = predict(regone,i)
                  elseif i in condrange2
                      ypred = predict(regtwo,i)
                  elseif i in condrange3
                      ypred = predict(regthree,i)
                  elseif i in condrange4
                      ypred = predict(regfour,i)
                  else
                      ypred = predict(regfive,i)
                  end
                  append!(e,ypred)
              end
              return(e)
      end
#==
Isotonic
    Regression
==#
@doc """
      FUNCTION NOT YET WRITTEN\n
      Isotonic Regression is used to predict continuous features with high
      variance.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.OneHotEncode(array)\n"""
mutable struct IsotonicRegression
    x
    y
end
function pred_isotonicregression(m,xt)
    if length(m.x) != length(m.y)
        throw(ArgumentError("The array shape does not match!"))
    end
end
#==
Multiple
    Linear
        Regression
==#
@doc """
      Multiple Linear Regression is used to influence LinearRegression with
      multiple features by averaging their predictions.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      model = Lathe.models.MultipleLinearRegression(x,y)\n"""
mutable struct MultipleLinearRegression
    x
    y
end
function pred_multiplelinearregression(m,xt)
    if length(m.x) != length(xt)
        throw(ArgumentError("Bad Feature Shape |
        Training Features are not equal!",))
    end
end
#==
Linear
    Regression
==#

@doc """
      Linear Regression is a well-known linear function used for predicting
      continuous features with a mostly linear or semi-linear slope.\n
      --------------------\n
      x = [7,6,5,6,5]\n
      y  = [3.4.5.6.3]\n
      xtrain = [7,5,4,5,3,5,7,8]\n
      model = Lathe.models.LinearRegression(x,y)
      y_pred = Lathe.models.predict(model,xtrain)\n
       """
mutable struct LinearRegression
    x
    y
end
#----  Callback
function pred_LinearRegression(m,xt)
    # a = ((∑y)(∑x^2)-(∑x)(∑xy)) / (n(∑x^2) - (∑x)^2)
    # b = (x(∑xy) - (∑x)(∑y)) / n(∑x^2) - (∑x)^2
    if length(m.x) != length(m.y)
        throw(ArgumentError("The array shape does not match!"))
    end
    # Get our x and y as easier variables
    x = m.x
    y = m.y
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
    xt = [i = a + (b * i) for i in xt]
    return(xt)
end
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
      Type:: Type determines which Linear Least Square algorithm to use,
      :LIN, :OLS, :WLS, and :GLS are the three options.\n
      - :LIN = Linear Least Square Regression\n
      - :OLS = Ordinary Least Squares\n
      - :WLS = Weighted Least Squares\n
      - :GLS = General Least Squares
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

#======================================================================
=======================================================================
            CATEGORICAL MODELS             CATEGORICAL MODELS
            CATEGORICAL MODELS             CATEGORICAL MODELS
======================================================================
======================================================================#
#==
Majority
    Class
        Baseline
==#
@doc """
      FUNCTION NOT YET WRITTEN\n
      Majority class baseline is used to find the most often interpreted
      classification in an array.\n
      --------------------\n
       """
function MajBaseline

end
#==
Multinomial
    Naive
        Bayes
==#
function MultinomialNB(x,y)

end
#==
Logistic
    Regression
==#
#==
@doc """
      One hot encoder replaces a single feature with sub arrays containing
      boolean values (1 or 0) for each individual category.\n
      --------------------\n
      array = [5,10,15]\n
      scaled_feature = Lathe.preprocess.OneHotEncode(array)\n
       """ ->
       ==#
function LogisticRegression(x,y)

end
#==
Nueral
    Network
        Framework
==#
#
#----------------------------------------------
end
