macro mu(x)
    x = eval(x)
    mean(x)
end
macro sigma(x)
    x = eval(x)
    std(x)
end
macro r(x,y)
    x = eval(x)
    y = eval(y)
    correlationcoeff(x,y)
end
macro t(samp,gen)
    samp = eval(samp)
    gen = eval(gen)
    independent_t(samp,gen)
end
macro f(samp,gen)
    samp = eval(samp)
    gen = eval(gen)
    f_test(samp,gen)
end
macro -(samp,gen)
    samp = eval(samp)
    gen = eval(gen)
    sign(samp,gen)
end
macro chi(samp,gen)
    samp = eval(samp)
    gen = eval(gen)
    chisq(samp,gen)
end
macro p(p,a,b)
    bay_ther(p,a,b)
end
macro acc(yhat,ytest)
    yhat = eval(yhat)
    ytest = eval(ytest)
    tp = yhat[1]
    if typeof(tp) == String
        catacc(ytest,yhat)
        print("we predicted acc")
    else
        sample, general = TrainTestSplit(yhat,.5)
        corr = @r sample general
        if length(unique(yhat)) == 2
            catacc(ytest,yhat)
            print("we predicted acc")
        else
            r2(ytest,yhat)
        end
    end

end
macro n(x)
    x = eval(x)
    length(x)
end
