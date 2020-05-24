#================
Model
    Validation
        Module
================#
@doc """
      |====== Lathe.validate ======\n
      |____________/ Metrics ___________\n
      |_____validate.mae(actual,pred)\n
      |_____validate.r2(actual,pred)\n
      |___________/ Feature-Selection ___________\n
      |_____validate.permutation(model)
       """
#-------Model Metrics--------____________
using Lathe
## <---- Mean Absolute Error ---->
@doc """
      Mean absolute error (MAE) subtracts two arrays and averages the
      difference.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """
function mae(actual,pred)
    l = length(actual)
    lp = length(pred)
    if l != lp
        throw(ArgumentError("The array shape does not match!"))
    end
    result = actual-pred
    maeunf = Lathe.stats.mean(result)
    if maeunf < 0
        maeunf = (maeunf - maeunf) - maeunf
    end
    return(maeunf)
end
# <---- Mean Squared Error ---->
@doc """
      Mean Square error (MSE) subtracts two arrays, squares the
      difference, and averages the result\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """
function mse(y,ŷ)
    diff = y .- ŷ
    diff = diff .^ 2
    Σdiff = sum(diff)
    return(Σdiff)
end
# <---- R Squared ---->
@doc """
      R squared is the correlation coefficient of regression, and is found
      by squaring the correlation coefficient.\n
      --------------------\n
      array = [5,10,15]\n
      r = Lathe.stats.anova(array)\n
       """
function r2(actual,pred)
    l = length(actual)
    lp = length(pred)
    if l != lp
        throw(ArgumentError("The array shape does not match!"))
    end
    r = Lathe.stats.correlationcoeff(actual,pred)
    rsq = r^2
    rsq = rsq * 100
    return(rsq)
end
#--------------------------------------------
# End
