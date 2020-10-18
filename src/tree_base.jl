function information_gain(group, den)
    res = 0.0
    for x in group
        p = x/den
        res -= p * log2(p)
    end
    return res
end

function information_impurity(information_before, left, right, ll, lr, lt)
    information_before -
    ll/lt * information_gain(left, ll) -
    lr/lt * information_gain(right, lr)
end

function gini_index(group, den)
    den == 0 && return 0.0
    res = 0.0
    for x in group
        res += (x/den)^2
    end
    return res
end

function gini_impurity(gini_before, left, right, ll, lr, lt)
    ll/lt * gini_index(left, ll) +
    lr/lt * gini_index(right, lr) -
    gini_before
end
mutable struct Node{T}
    feature_idx::Int
    feature_val::T
    value::Int
    left::Node{T}
    right::Node{T}
    is_terminal::Bool

    function Node(feature_idx, feature_val::T) where {T}
        node = new{T}()
        node.feature_idx = feature_idx
        node.feature_val = feature_val
        node.is_terminal = false

        return node
    end

    function Node{T}(value) where {T}
        node = new{T}()
        node.value = value
        node.is_terminal = true

        return node
    end

    function Node(feature_idx, feature_val::T, value, is_terminal=false) where {T}
        node = new{T}()
        node.feature_idx = feature_idx
        node.feature_val = feature_val
        node.value = value
        node.is_terminal = is_terminal

        return node
    end

    function Node{T}() where T
        node = new{T}()
        node.is_terminal = false
        return node
    end
end

struct DecisionTreeContainer{T}
    root::Node{T}
    n_features_per_node::Int
    n_classes::Int
    max_depth::Int
    min_node_records::Int
end

"""
    feature_best_split
For a given feature search best split value.
"""
function feature_best_split(containers, X, y, n_classes, feature)
    gini_before = containers.gini_before
    left = containers.left
    right = containers.right
    lt = containers.lt

    # prepare initial split
    left .= 0
    right .= 0
    for i in axes(X, 1)
        right[y[i]] += 1
    end
    # TODO: I leave it for now. In the future, sorting should be done before feature split,
    # so allocations or unsafe arrays would be of no importance.
    sort_idx = sortperm(@view X[:, feature])
    ll = 1
    lr = length(y) - 1
    i1 = sort_idx[1]
    left[y[i1]] = 1
    right[y[i1]] -= 1
    prev_val = X[i1, feature]
    best_val = prev_val
    best_impurity = 0.0
    @inbounds for idx in 2:length(y)
        i = sort_idx[idx]
        if X[i, feature] != prev_val
            prev_val = X[i, feature]
            impurity = gini_impurity(gini_before, left, right, ll, lr, lt)
            if impurity > best_impurity
                best_impurity = impurity
                best_val = prev_val
            end
        end
        ll += 1
        lr -= 1
        left[y[i]] += 1
        right[y[i]] -= 1
    end

    return (val = best_val, impurity = best_impurity)
end

function create_containers(n_classes, y)
    left = zeros(Int, n_classes)
    right = Vector{Int}(undef, n_classes)
    lt = length(y)
    for i in 1:lt
        left[y[i]] += 1
    end
    gini_before = gini_index(left, lt)
    containers = (left = left, right = right, gini_before = gini_before, lt = lt)

    return containers
end

# Chooses best feature from features
function best_split(X, target, n_classes, features)
    containers = create_containers(n_classes, target)
    best_feature = 0
    best_val = -Inf
    best_impurity = -Inf
    for feature in features
        val, impurity = feature_best_split(containers, X, target, n_classes, feature)
        if impurity > best_impurity
            best_val = val
            best_feature = feature
            best_impurity = impurity
        end
    end

    return (feature = best_feature, val = best_val)
end

function split_value(X, target, n_classes)
    res = zeros(Int, n_classes)
    for i in axes(X, 1)
        res[target[i]] += 1
    end

    return argmax(res)
end

function get_split_indices(X, feature_idx, feature_val)
    return X[:, feature_idx] .< feature_val, X[:, feature_idx] .>= feature_val
end

function is_pure(target)
    return all(target[1] .== target)
end

###############################
# Node functions
###############################

function process_node(dtc::DecisionTreeContainer{T}, node, X, target,
                      rng = Random.GLOBAL_RNG,
                      features = sample(rng, 1:size(X, 2), dtc.n_features_per_node, replace = false),
                      depth = 1) where T

    if depth > dtc.max_depth
        node.is_terminal = true
        node.value = split_value(X, target, dtc.n_classes)
    elseif length(target) <= dtc.min_node_records
        node.is_terminal = true
        node.value = split_value(X, target, dtc.n_classes)
    elseif is_pure(target)
        node.is_terminal = true
        node.value = target[1]
    else
        feature_idx, feature_val = best_split(X, target, dtc.n_classes, features)
        node.feature_idx = feature_idx
        node.feature_val = feature_val
        left_ids, right_ids = get_split_indices(X, feature_idx, feature_val)
        left = Node{T}()
        right = Node{T}()
        node.left = left
        node.right = right
        new_features = sample(rng, 1:size(X, 2), dtc.n_features_per_node, replace = false)
        process_node(dtc, left, X[left_ids, :], target[left_ids], rng, new_features, depth + 1)
        process_node(dtc, right, X[right_ids, :], target[right_ids], rng, new_features, depth + 1)
    end
end

function create_tree(X, y; rng = Random.GLOBAL_RNG, max_depth = 10, min_node_records = 1,
                     n_features = size(X, 2))
    T = eltype(X)
    root = Node{T}()
    n_classes = length(Set(y))
    dtc = DecisionTreeContainer(root, n_features, n_classes, max_depth, min_node_records)
    process_node(dtc, root, X, y, rng)

    return root
end

function predict(node::Node, row)
    if node.is_terminal
        return node.value
    else
        if row[node.feature_idx] < node.feature_val
            return predict(node.left, row)
        else
            return predict(node.right, row)
        end
    end
end

function predict(node::Node, X, i)
    if node.is_terminal
        return node.value
    else
        if X[i, node.feature_idx] < node.feature_val
            return predict(node.left, X, i)
        else
            return predict(node.right, X, i)
        end
    end
end
