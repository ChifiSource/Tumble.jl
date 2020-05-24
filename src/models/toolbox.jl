#==
Power Log
==#
@doc """
      A powerlog can be used to perform a one-tailed test, as well as get the proper sample size for a testing population.\n
      --------------------\n
      ==PARAMETERS==\n
     p1 <- A Float64 percentage representing the probability of scenario one.\n
     p2 <- A Float64 percentage representing the probability of scenario two. These two probability values should follow these guidelines: p1 = p1 + x = p2\n
     alpha = 0.05 <- Sets an alpha value\n
     --------------------\n
     Returns power, sample_size
       """
function PowerLog(p1::Float64,p2::Float64; alpha::Float64 = 0.05, rsq::Real = 0)
    pd = p2 - p1
    l1 = p1/(1-p1)
    l2 = p2/(1-p2)
    θ = l2 / l1
    or = θ
    λ = log(θ)
    λ2 = λ^2
    za = quantile(normal(),1-alpha)
    println("One-tailed test: alpha = ",alpha,", p1 = ",p1,", p2 = ",p2,", rsq = ",rsq,", odds ratio = ",or)
    δ = (1 + (1 + λ2)*exp(5 * λ2/4))/(1 + exp(-1*λ2/4))
    pwr = zeros(Float64,8)
    nn = zeros(Int64,8)
    i = 1
    for power = 0.6:.05:.95
        zb = quantile(normal(),power)

        N = ((za + zb*exp(-1 * λ2/4))^2 * (1 + 2*p1*δ))/(p1*λ2)
        N /= (1 - rsq)
        pwr[i] = power
        nn[i] = ceil(Int64,N)
        i += 1
    end
    return(pwr, nn)
end
