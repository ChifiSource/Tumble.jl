#===============================
<-----------Lathe.jl----------->
Programmed by Emmett Boudreau
    <emmett@emmettboudreau.com>
        <http://emmettboudreau.com>
GNU General Open Source License
    (V 3.0.0)
        Free for Modification and
        Redistribution
Thank you for your forks!
<-----------Lathe.jl----------->
================================#
module Lathe
using DataFrames
#================
Stats
    Module
================#
module stats
#<----Mean---->
function mean(array)
    observations = length(array)
    average = sum(array)/observations
    return(average)
end
#<----Variance---->
function variance(array)
    mean = mean(array)
    sq = sum(array) - mean
    squared_mean = expo(sq,2)
    return(squared_mean)
end
#<----Standard Deviation---->
function standardize(array)
    mean = sum(array)/length(array)
    sq = sum(array) - mean
    squared_mean = expo(sq,2)
    standardized = sqrt(squared_mean)
    return(standardized)
end
#<----Exponential Expression---->
function expo(number,scalar)
    if scalar != 0
        newscale = scalar-1
        newnumber = number * number
        expo(newnumber,newscale)
    else
        return(number)
    end
end
#<----T Value---->
function t_test(sample,general)
    sampmean = mean(sample)
    genmean = mean(general)
    samples = length(sample)
    std = standardize(general)
    t = (sampmean - genmean) / (std / sqrt(samples))
    return(t)
end
#<----Bayes Theorem---->
#P = prob, A = prior, B = Evidence,
function bay_ther(p,a,b)
    psterior = (p*(b|a) * p*(a)) / (p*b)
    return(psterior)
end
#<----Confidence Intervals---->
function confiint(data, confidence=.95)
    n = length(data)
    mean = sum(data)/n
    #stderr = scs.sem(data)
    #interval = stderr * scs.t.ppf((1 + confidence) / 2.0, n-1)
    #return (mean , mean-interval, mean+interval)
end
#<----Standard Error---->
function standarderror(data)
    std = standardize(data)
    sample = length(data)
    ste = (std/sqrt(sample))
    return(ste)
end
end
#================
Preprocessing
     Module
================#
module preprocess
#hello
end
#================
Predictive
    Learning
        Models
================#
module predmodel
#==
import .Trees: TabularData, Tree, Node, Leaf, Criterion, Gini, CrossEntropy
import MSE, getroot, getleft, getright, isnode, n_samples, impurity
#==
Tree
    Models
    ==#
#<---- Trees Base ---->
module Trees

using StatsBase
using DataFrames

include("example.jl")
include("sort.jl")
include("split.jl")

export Tree, fit, predict

abstract Element

type Node{T} <: Element
    feature::Int
    threshold::T
    impurity::Float64
    n_samples::Int
    left::Int
    right::Int
end

abstract Leaf <: Element

type ClassificationLeaf <: Leaf
    counts::Vector{Int}
    impurity::Float64
    n_samples::Int

    function ClassificationLeaf(example::Example, samples::Vector{Int}, impurity::Float64)
        counts = zeros(Int, example.n_labels)
        for s in samples
            label = example.y[s]
            counts[label] += round(Int, example.sample_weight[s])
        end
        new(counts, impurity, length(samples))
    end
end

majority(leaf::ClassificationLeaf) = indmax(leaf.counts)

type RegressionLeaf <: Leaf
    mean::Float64
    impurity::Float64
    n_samples::Int

    function RegressionLeaf(example::Example, samples::Vector{Int}, impurity::Float64)
        new(mean(example.y[samples]), impurity, length(samples))
    end
end

Base.mean(leaf::RegressionLeaf) = leaf.mean

immutable Undef <: Element; end
const undef = Undef()

# parameters to build a tree
immutable Params
    criterion::Criterion
    splitter  # splitter constructor
    max_features::Int
    max_depth::Int
    min_samples_split::Int
end

# bundled arguments for splitting a node
immutable SplitArgs
    index::Int
    depth::Int
    range::UnitRange{Int}
end

type Tree
    index::Int
    nodes::Vector{Element}

    Tree() = new(0, Element[])
end

getnode(tree::Tree, index::Int) = tree.nodes[index]
getroot(tree::Tree) = getnode(tree, 1)
getleft(tree::Tree, node::Node) = getnode(tree, node.left)
getright(tree::Tree, node::Node) = getnode(tree, node.right)

isnode(tree::Tree, index::Int) = isa(tree.nodes[index], Node)
isleaf(tree::Tree, index::Int) = isa(tree.nodes[index], Leaf)
isnode(node::Element) = isa(node, Node)
isleaf(node::Element) = isa(node, Leaf)

impurity(node::Node) = node.impurity
impurity(leaf::Leaf) = leaf.impurity
n_samples(node::Node) = node.n_samples
n_samples(leaf::Leaf) = leaf.n_samples

function next_index!(tree::Tree)
    push!(tree.nodes, undef)
    tree.index += 1
end

function fit(tree::Tree, example::Example, criterion::Criterion, max_features::Int, max_depth::Int, min_samples_split::Int)
    if isa(criterion, ClassificationCriterion)
        splitter = ClassificationSplitter{typeof(criterion)}
    elseif isa(criterion, RegressionCriterion)
        splitter = RegressionSplitter{typeof(criterion)}
    else
        error("invalid criterion")
    end
    params = Params(criterion, splitter, max_features, max_depth, min_samples_split)
    samples = where(example.sample_weight)
    sample_range = 1:length(samples)
    next_index!(tree)
    args = SplitArgs(tree.index, 1, sample_range)
    build_tree(tree, example, samples, args, params)
    return
end

function where(v::AbstractVector)
    n = countnz(v)
    indices = Array(Int, n)
    i = 1
    j = 0

    while (j = findnext(v, j + 1)) > 0
        indices[i] = j
        i += 1
    end

    indices
end

function leaf(example::Example, samples, criterion::RegressionCriterion)
    RegressionLeaf(example, samples, impurity(samples, example, criterion))
end

function leaf(example::Example, samples, criterion::ClassificationCriterion)
    ClassificationLeaf(example, samples, impurity(samples, example, criterion))
end

function build_tree(tree::Tree, example::Example, samples::Vector{Int}, args::SplitArgs, params::Params)
    n_features = example.n_features
    range = args.range  # shortcut
    n_samples = length(range)

    if args.depth >= params.max_depth || n_samples < params.min_samples_split
        tree.nodes[args.index] = leaf(example, samples[range], params.criterion)
        return
    end

    best_feature = 0
    best_impurity = Inf
    local best_threshold, best_boundary

    for k in sample(1:n_features, params.max_features, replace=false)
        feature = example.x[samples[range], k]
        sort!(samples, feature, range)
        splitter = params.splitter(samples, feature, range, example)

        for s in splitter
            averaged_impurity = (s.left_impurity * s.n_left_samples + s.right_impurity * s.n_right_samples) / (s.n_left_samples + s.n_right_samples)

            if averaged_impurity < best_impurity
                best_impurity = averaged_impurity
                best_feature = k
                best_threshold = s.threshold
                best_boundary = s.boundary
            end
        end
    end

    if best_feature == 0
        tree.nodes[args.index] = leaf(example, samples[range], params.criterion)
    else
        feature = example.x[samples[range], best_feature]
        sort!(samples, feature, range)

        left = next_index!(tree)
        right = next_index!(tree)
        tree.nodes[args.index] = Node(best_feature, best_threshold, best_impurity, n_samples, left, right)

        next_depth = args.depth + 1
        left_node = SplitArgs(left, next_depth, range[1:best_boundary])
        right_node = SplitArgs(right, next_depth, range[best_boundary+1:end])
        build_tree(tree, example, samples, left_node, params)
        build_tree(tree, example, samples, right_node, params)
    end

    return
end

function predict(tree::Tree, x::AbstractVector)
    node = getroot(tree)

    while true
        if isa(node, Node)
            if x[node.feature] <= node.threshold
                # go left
                node = getleft(tree, node)
            else
                # go right
                node = getright(tree, node)
            end
        elseif isa(node, ClassificationLeaf)
            return majority(node)
        elseif isa(node, RegressionLeaf)
            return mean(node)
        else
            error("found invalid type of node (type: $(typeof(node)))")
        end
    end
end

end  # module Tree
#<---- Random Forests ---->
    # learner
    learner::Union{T,Void}

    function RandomForest(n_estimators, max_features, max_depth, min_samples_split, criterion)
        if n_estimators < 1
            error("You need at least one estimater! (got: $n_estimators)")
        end

        if isa(max_features, Integer) && max_features < 1
            error("You need at least one feature! (got: $max_features)")
        elseif isa(max_features, AbstractFloat) && !(0. < max_features <= 1.)
            error("You need at least one feature! (got: $max_features)")
        elseif isa(max_features, Symbol) && !(is(max_features, :sqrt) || is(max_features, :third))
            error("max_features should be :sqrt or :third (got: $max_features)")
        end

        if is(max_depth, nothing)
            max_depth = typemax(Int)
        elseif isa(max_depth, Integer) && max_depth <= 1
            error("You need a max death of at least one! (got: $max_depth)")
        end

        if min_samples_split <= 1
            error("sample split must be at least one (got: $min_samples_split)")
        end

        if is(criterion, :gini)
            criterion = Gini
        elseif is(criterion, :entropy)
            criterion = CrossEntropy
        elseif is(criterion, :mse)
            criterion = MSE
        else
            error("Invalid criteria! (got: $criterion)")
        end

        new(n_estimators, max_features, max_depth, min_samples_split, criterion, nothing)
    end
end

function resolve_max_features(max_features::Any, n_features::Int)
    if is(max_features, :sqrt)
        floor(Int, sqrt(n_features))
    elseif is(max_features, :third)
        div(n_features, 3)
    elseif isa(max_features, Integer)
        max(max_features, n_features)
    elseif isa(max_features, AbstractFloat)
        floor(Int, n_features * max_features)
    elseif is(max_features, nothing)
        n_features
    else
        error("Bad Max Features; $max_features")
    end
end

function feature_importances(rf::RandomForest)
    if is(rf.learner, nothing)
        error("Model not trained!")
    end
    rf.learner.improvements
end

function set_improvements!(learner)
    improvements = learner.improvements

    for tree in learner.trees
        root = getroot(tree)
        add_improvements!(tree, root, improvements)
    end
    normalize!(improvements)
end

function add_improvements!(tree::Tree, node::Node, improvements::Vector{Float64})
    left = getleft(tree, node)
    right = getright(tree, node)
    n_left_samples = n_samples(left)
    n_right_samples = n_samples(right)
    averaged_impurity = (impurity(left) * n_left_samples + impurity(right) * n_right_samples) / (n_left_samples + n_right_samples)
    improvement = impurity(node) - averaged_impurity
    improvements[node.feature] += improvement

    add_improvements!(tree, left, improvements)
    add_improvements!(tree, right, improvements)
    return
end

function add_improvements!(::Tree, ::Leaf, ::Vector{Float64})
    # do nothing!
    return
end
==#
end
#================
Processing
    Models
================#
#==
This is the end of the main
module, nothing is to be written
beyond here
==#
end
