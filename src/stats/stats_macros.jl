"""
      This macro calls Lathe.stats: mean. It takes a single array as a
          parameter and returns the mean of said array.\n
      --------------------\n
      array = [5,10,15]\n
      mean = @mu array\n
      println(mean)\n
        10
       """
macro mu(x)
    x = eval(x)
    mean(x)
end
"""
      This macro calls Lathe.stats: std. It takes a single array as a
          parameter and returns the standard deviation of said array.\n
      --------------------\n
      array = [5,10,15]\n
      std = @std array\n
       """
macro sigma(x)
    x = eval(x)
    std(x)
end
"""
      This macro calls Lathe.stats: correlationcoeff. It takes a sample and
          the population as a parameter and will return r.\n
      --------------------\n
      array = [5,10,15]\n
      samparray = [5,10,15]\n
      r = @r samparray array\n
       """
macro r(x,y)
    x = eval(x)
    y = eval(y)
    correlationcoeff(x,y)
end
"""
      This macro calls Lathe.stats: bay_ther. It takes probability, prior, and
          evidence as parameters parameter and returns a posterier.\n
      --------------------\n
      prob = .20\n
      prior = .10\n
      evidence = .3
      p = @p prob prior evidence\n
       """
macro p(p,a,b)
    bay_ther(p,a,b)
end
"""
      EXPERIMENTAL -- This macro will give accuracy for both continuous and
      categorical features by predicting whether a given model is categorical or
      not. In the event that categorical targets are predicted, it will use
      Lathe.stats: catacc (from Lathe.stats.validate.catacc), in the event
      that it predicts a continuous target, it will calculate accuracy using
      Lathe.stats: r2 (from Lathe.stats.validate.r2).\n
      --------------------\n
      test_y = [1,0,1,0]\n
      yhat = [0,1,1,0]\n
      @acc yhat test_y\n
      .5
       """
macro acc(yhat,ytest)
    yhat = eval(yhat)
    ytest = eval(ytest)
    tp = yhat[1]
    if typeof(tp) == String
        catacc(ytest,yhat)
        print("we predicted acc")
    else
        if length(unique(yhat)) == 2
            catacc(ytest,yhat)
            print("we predicted acc")
        else
            r2(ytest,yhat)
        end
    end

end
"""
      This macro is simply a symbol usage to get the length of an array.\n
      --------------------\n
      array = [5,10,15]\n
      @n array\n
      3
       """
macro n(x)
    x = eval(x)
    length(x)
end
