"""
    ## Phi
    ### Description
     Phi, the totient function counts the positive numbers
     that are prime or relatively prime up to n.\n
      --------------------\n
    ### Input
      bay_ther(p, a, b)\n
      --------------------\n
      #### Positional Arguments
      Array{Any} - n:: The sample for the function to run on.\n
      --------------------\n
     ### Output
     φ:: Sum of all relatively prime numbers in a given array.
       """
φ(n) = sum(1 for k in 1:n if gcd(n, k) == 1)
