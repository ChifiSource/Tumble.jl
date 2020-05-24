#==
Nueral
    Network
        Framework
==#
function calculate_activation_forward(A_pre , W , b , function_type)
    if (function_type == "sigmoid")
        Z , linear_step_cache = forward_linear(A_pre , W , b)
        A , activation_step_cache = sigmoid(Z)
    elseif (function_type == "relu")
        Z , linear_step_cache = forward_linear(A_pre , W , b)
        A , activation_step_cache = relu(Z)
    end
    cache = (linear_step_cache , activation_step_cache)
    return A , cache
end

function model_forward_step(X , params)
    all_caches = []
    A = X
    L = length(params)/2
    for l=1:L-1
        A_pre = A
        A , cache = calculate_activation_forward(A_pre , params[string("W_" , string(Int(l)))] , params[string("b_" , string(Int(l)))] , "relu")
        push!(all_caches , cache)
    end
    A_l , cache = calculate_activation_forward(A , params[string("W_" , string(Int(L)))] , params[string("b_" , string(Int(L)))] , "sigmoid")
    push!(all_caches , cache)
    return A_l , all_caches
end
function cost_function(AL , Y)
    cost = -mean(Y.*log.(AL) + (1 .- Y).*log.(1 .- AL))
    return cost
end
function forward_linear(A , w , b)
    Z = w*A .+ b
    cache = (A , w , b)
    return Z,cache
end
function init_param(layer_dimensions)
    param = Dict()
    for l=0:length(layer_dimensions)
        param[string("W_" , string(l-1))] = rand(layer_dimensions[l] ,
				layer_dimensions[l-1])*0.1
        param[string("b_" , string(l-1))] = zeros(layer_dimensions[l] , 1)
    end
    return param
end

function backward_linear_step(dZ , cache)
    A_prev , W , b = cache
    m = size(A_prev)[2]
    dW = dZ * (A_prev')/m
    db = sum(dZ , dims = 2)/m
    dA_prev = (W')* dZ
    return dW , db , dA_prev
end
function relu(X)
    rel = max.(0,X)
    return rel , X
end
function backward_relu(dA , cache_activation)
    return dA.*(cache_activation.>0)
end
function backward_sigmoid(dA , cache_activation)
    return dA.*(sigmoid(cache_activation)[1].*(1 .- sigmoid(cache_activation)[1]))
end
function backward_activation_step(dA , cache , activation)
    linear_cache , cache_activation = cache
    if (activation == "relu")
        dZ = backward_relu(dA , cache_activation)
        dW , db , dA_prev = backward_linear_step(dZ , linear_cache)
    elseif (activation == "sigmoid")
        dZ = backward_sigmoid(dA , cache_activation)
        dW , db , dA_prev = backward_linear_step(dZ , linear_cache)
    end
    return dW , db , dA_prev

end
function model_backwards_step(A_l , Y , caches)
    grads = Dict()
    L = length(caches)
    m = size(A_l)[2]
    Y = reshape(Y , size(A_l))
    dA_l = (-(Y./A_l) .+ ((1 .- Y)./( 1 .- A_l)))
    current_cache = caches[L]
    grads[string("dW_" , string(L))] , grads[string("db_" , string(L))] , grads[string("dA_" , string(L-1))] = backward_activation_step(dA_l , current_cache , "sigmoid")
    for l=reverse(0:L-2)
        current_cache = caches[l+1]
        grads[string("dW_" , string(l+1))] , grads[string("db_" , string(l+1))] , grads[string("dA_" , string(l))] = backward_activation_step(grads[string("dA_" , string(l+1))] , current_cache , "relu")

    end
    return grads
end

function update_param(parameters , grads , learning_rate)
    L = Int(length(parameters)/2)
    for l=0:(L-1)
        parameters[string("W_" , string(l+1))] -= learning_rate.*grads[string("dW_" , string(l+1))]
        parameters[string("b_",string(l+1))] -= learning_rate.*grads[string("db_",string(l+1))]
    end
    return parameters
end
@doc """
      'Network' specifically constructs a convolutional nueral network,\n
      though this network is still being worked on in both methodology\n
      and in a physical manner,\n thanks for understanding.
       """
function Network(X,Y,layers_dimensions,n_iter)
    params = init_param(layers_dimensions)
    costs = []
    iters = []
    accuracy = []
    for i=1:n_iter
        A_l , caches  = model_forward_step(X , params)
        cost = cost_function(A_l , Y)
        acc = check_accuracy(A_l , Y)
        grads  = model_backwards_step(A_l , Y , caches)
        params = update_param(params , grads , learning_rate)
        println("Iteration ->" , i)
        println("Cost ->" , cost)
        println("Accuracy -> " , acc)
        push!(iters , i)
        push!(costs , cost)
        push!(accuracy , acc)
    end
    predict(xt) = (xt + 4)
    (test)->(costs;params;predict)
end
