#================
Model
    Validation
        Module
================#
#-------Model Metrics--------____________
"""
      Mean absolute error (MAE) subtracts two arrays and averages the
      difference.\n
      --------------------\n
      actual = [5,10,15,20]\n
      pred = [5,10,15,20]\n
      mae(actual,pred)\n
       """
function mae(actual,pred)
    l = length(actual)
    lp = length(pred)
    if l != lp
        throw(ArgumentError("The array shape does not match!"))
    end
    result = actual-pred
    maeunf = mean(result)
    if maeunf < 0
        maeunf = (maeunf - maeunf) - maeunf
    end
    return(maeunf)
end
# <---- Mean Squared Error ---->
"""
      Mean Square error (MSE) subtracts two arrays, squares the
      difference, and averages the result\n
      --------------------\n
      actual = [5,10,15]\n
      pred = [5,10,15]
      r = mse(actual, pred)\n
       """
function mse(y,ŷ)
    diff = y .- ŷ
    diff = diff .^ 2
    Σdiff = sum(diff)
    return(Σdiff)
end
# <---- R Squared ---->
"""
      R squared is the correlation coefficient of regression, and is found
      by squaring the correlation coefficient.\n
      --------------------\n
      actual = [5,10,15]\n
      pred = [5,10,15]\n
      r2 = r2(actual, pred)\n
       """
function r2(actual,pred)
    l = length(actual)
    lp = length(pred)
    if l != lp
        throw(ArgumentError("The array shape does not match!"))
    end
    r = correlationcoeff(actual,pred)
    rsq = r^2
    rsq = rsq * 100
    return(rsq)
end
"""
Categorical accuracy is used to determine the accuracy of a categorical model
by calculating the percentage of correct predictions.\n
------------------\n
testy = ["hi","hello"]\n
yhat = ["hi","hi"]\n
accuracy = catacc(yhat,testy)\n
.5
    """
function catacc(testy,yhat)
    n = length(yhat)
    if n != length(testy)
        throw(ArgumentError("The array shape does not match!"))
    end
    score = 0
    for (i,w) in zip(yhat,testy)
        if i == w
            score += 1
        end
    end
    acc = score / n
end
#--------------------------------------------
# End
