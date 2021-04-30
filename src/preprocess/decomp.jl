
mutable struct SVD{P} <: Transformer
    predict::P
    iter::Int64
    function SVD(maxiter=1000)
        predict(A, r) = -svd(A, r, maxiter = iter)
        iter = maxiter
        new{typeof(predict)}(predict, iter)
    end
    function _svd(A, r; iter = 100)
        V = randn(size(A,2),r) # random initialization
        for _ = 1:maxiter
            W = A * V
            Z = A' * W
            V, R = mmids_gramschmidt(Z)
        end
        W = A * V
        S = [norm(W[:, i]) for i=1:size(W,2)] # singular values
        U = reduce(hcat,[W[:,i]/S[i] for i=1:size(W,2)]) # left singular vectors
        return U * S * V'
end
end
#===
Random
Projection
Trees
===#
struct RandomProjectionTreeNode{T <: Number,
                                V <: AbstractVector{T}}
    indices
    isleaf::Bool
    hyperplane::Union{V, Nothing}
    offset::Union{T, Nothing}
    leftchild::Union{RandomProjectionTreeNode{T, V}, Nothing}
    rightchild::Union{RandomProjectionTreeNode{T, V}, Nothing}
end


struct RandomProjectionTree{T <: Number, V <: AbstractVector{T}}
    root::RandomProjectionTreeNode{T, V}
    leafsize
end

"""
    RandomProjectionTree(data; leafsize = 30)
A RandomProjectionTree is a binary tree whose non-leaf nodes correspond to random
hyperplanes in ℜᵈ and whose leaf nodes represent subregions that partition this
space.
"""
function RandomProjectionTree(data; leafsize = 30)
    root = build_rptree(data, 1:length(data), leafsize)
    return RandomProjectionTree(root, leafsize)
end


"""
    build_rptree(data::AbstractVector, indices, leafsize) -> RandomProjectionTreeNode
Recursively construct the RP tree by randomly splitting the data into nodes.
"""
function build_rptree(data::U, indices, leafsize) where {T <: Number,
                                                         V <: AbstractVector{T},
                                                         U <: AbstractVector{V}}
    if length(indices) <= leafsize
        return RandomProjectionTreeNode{T, V}(indices,
                                              true,
                                              nothing,
                                              nothing,
                                              nothing,
                                              nothing)
    end

    leftindices, rightindices, hyperplane, offset = rp_split(data, indices)
    leftchild = build_rptree(data, leftindices, leafsize)
    rightchild = build_rptree(data, rightindices, leafsize)

    return RandomProjectionTreeNode(nothing,
                                    false,
                                    hyperplane,
                                    offset,
                                    leftchild,
                                    rightchild)
end


"""
    search_tree(tree, point) -> node::RandomProjectionTreeNode
Search a random projection tree for the node to which a point belongs.
"""
function search_rptree(tree::RandomProjectionTree{T, V},
                       point::V
                       ) where {T, V}

    node = tree.root
    while !node.isleaf
        node = select_side(node.hyperplane, node.offset, point) ? node.leftchild : node.rightchild
    end
    return node

end

#############################
# Random Projection Forests
#############################

struct RandomProjectionForest{T, V}
    trees::Vector{RandomProjectionTree{T, V}}
end

"""
    RandomProjectionForest(data, args...; n_trees, kwargs...) -> RandomProjectionForest
Create a collection of `n_trees` random projection trees built using `data`.
`args...` and `kwargs...` are passed to the RandomProjectionTree constructor.
"""
function RandomProjectionForest(data, args...;
                                n_trees, kwargs...)
    # TODO: threading
    trees = [RandomProjectionTree(data, args...; kwargs...) for _ in 1:n_trees]
    return RandomProjectionForest(trees)
end

"""
    approx_knn(tree, data, point, n_neighbors) -> indices, distances
Find approximate nearest neighbors to `point` in `data` using a random
projection tree built on this data. Returns the indices and distances to the
approximate knns as arrays, sorted by distance.
"""
function approx_knn(tree::RandomProjectionTree,
                    data,
                    point,
                    n_neighbors)
    # TODO: handle case when n_neighbors > tree.leafsize
    candidates = search_rptree(tree, point).indices

    return _approx_knn(data, point, candidates, n_neighbors)
end

"""
"""
function approx_knn(forest::RandomProjectionForest,
                    data,
                    point,
                    n_neighbors)
    # TODO: threading
    candidates = []
    for tree in forest.trees
        union!(candidates, search_rptree(tree, point).indices)
    end
    return _approx_knn(data, point, candidates, n_neighbors)
end

"""
"""
function _approx_knn(data, point, candidates, n_neighbors)
    length(candidates) >= n_neighbors || @warn "Fewer candidates than n_neighbors!"
    # TODO: threading
    distances = [norm(data[i] - point) for i in candidates]
    perm = sortperm(distances)
    indices = candidates[perm]
    distances = distances[perm]

    if n_neighbors < length(indices)
        indices = indices[1:n_neighbors]
        distances = distances[1:n_neighbors]
    end
    return indices, distances
end



"""
    rp_split(data, indices) -> lefts, rights, hyperplane, offset
Splits the data into two sets depending on which side of a random hyperplane
each point lies.
Returns the left indices, right indices, hyperplane, and offset.
"""
function rp_split(data, indices)

    # select two random points and set the hyperplane between them
    # TODO: threading
    # TODO: remove StatsBase dep
    leftidx, rightidx = sample(indices, 2; replace=false)
    hyperplane = data[leftidx] - data[rightidx]

    offset = -adjoint(hyperplane) * (data[leftidx] + data[rightidx]) / 2

    lefts = sizehint!(Int32[], div(length(data), 2))
    rights = sizehint!(Int32[], div(length(data), 2))
    # TODO: threading
    for (i, idx) in enumerate(indices)
        # for each data vector, compute which side of the hyperplane
        select_side(hyperplane, offset, data[idx]) ? append!(lefts, idx) : append!(rights, idx)
    end

    return lefts, rights, hyperplane, offset
end

"""
Project `point` onto `hyperplane` and return `true` if it lies to the "left" of
`offset`, otherwise return `false`.
If the point is less than `√eps`, pick a side randomly.
"""
function select_side(hyperplane, offset, point)
    margin = adjoint(hyperplane) * point + offset
    if abs(margin) <= sqrt(eps(margin))
        # TODO: threading
        return rand() < .5
    end
    return margin < 0
end

Base.sizeof(t::RandomProjectionTree) = sizeof(t.root)

Base.sizeof(n::RandomProjectionTreeNode) = sizeof(n.indices) + sizeof(n.isleaf) +
                                           sizeof(n.hyperplane) + sizeof(n.offset) +
                                           sizeof(n.leftchild) + sizeof(n.rightchild)

num_nodes(t::RandomProjectionTree) = num_nodes(t.root)

function num_nodes(n::RandomProjectionTreeNode)
    n.isleaf || true + num_nodes(n.leftchild) + num_nodes(n.rightchild)
end

num_leaves(t::RandomProjectionTree) = num_leaves(t.root)

function num_leaves(n::RandomProjectionTreeNode)
    n.isleaf || num_leaves(n.leftchild) + num_leaves(n.rightchild)
end

leaves(t::RandomProjectionTree) = leaves(t.root)

function leaves(n::RandomProjectionTreeNode)
    n.isleaf && return [n]
    return append!(leaves(n.leftchild), leaves(n.rightchild))
end

max_nnz(t::RandomProjectionTree) = max_nnz(t.root)

function max_nnz(n::RandomProjectionTreeNode)
    n.isleaf && return 0
    return max(nnz(n.hyperplane), max_nnz(n.leftchild), max_nnz(n.rightchild))
end
