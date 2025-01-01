"""
DataConformingDeePC.jl is a Julia package implementing the data-conforming version of the DeePC algorithm 
in our paper "Floodgates up to contain the DeePC and limit extrapolation"
"""
module DataConformingDeePC

# ==== imports ====
using Convex, SCS, LinearAlgebra, Statistics

# ==== exports ====
export run_DeePC, DeePC_struct, ConstraintMatrices

"""
    ConstraintMatrices
Stores the constraint matrices representing polyhedron constraints of the form:
`Ay * y <= by` and `Au * u <= bu`.
"""
mutable struct ConstraintMatrices
    Ay::Union{Matrix{Float64}, Nothing}
    by::Union{Vector{Float64}, Nothing}
    Au::Union{Matrix{Float64}, Nothing}
    bu::Union{Vector{Float64}, Nothing}
end
"""
    DeePC_struct
Used to formulate the DeePC optimization problem
Q:                  the weighting matrix of the output signal
R:                  the weighting matrix of the input signal
UpYp:     the left tall matrix [Uₚ; Yₚ] in the paper
Uf:  Uf according to the paper
Yf: Yf according to the paper
Constrained:        true if the system is input or output constrained
constraintMatrices: params of polyhedral constraints
"""
mutable struct DeePC_struct
    # Data and parameters required to run DeePC
    Q::Matrix{Float64}
    R::Matrix{Float64}
    N::Int # Horizon length
    Tᵢₙᵢ::Int # Data length to account for initial conditions
    UpYp::Matrix{Float64} # [Uₚ; Yₚ]
    Uf::Matrix{Float64}
    Yf::Matrix{Float64}
    H_data_inv::Matrix{Float64}
    Constrained::Bool
    constraintMatrices::ConstraintMatrices

    function DeePC_struct(Q::Matrix{Float64}, R::Matrix{Float64}, 
                            N::Int, Ud::Matrix{Float64}, Yd::Matrix{Float64},
                            Tᵢₙᵢ::Int, Constrained::Bool)
        constraintMatrices = ConstraintMatrices(nothing, nothing, nothing, nothing)
        UpYp, Uf, Yf, H_data = build_data_matrix(Ud, Yd, Tᵢₙᵢ, N)
        H_data_inv = inv(H_data)
        H_data_inv = (H_data_inv + H_data_inv') ./ 2
        m, T = size(Ud)
            if T< (m+1) * (Tᵢₙᵢ + N + 1) - 1
                error("Length of training data (T) needs to be bigger to at least detect a 1st order model.")
            end
        new(Q, R, N, Tᵢₙᵢ, UpYp, Uf, Yf, H_data_inv, Constrained, constraintMatrices)        
    end
end

"""
    run_DeePC(prob::DeePC_struct, Uᵢₙᵢ::Vector{Float64}, Yᵢₙᵢ::Vector{Float64}; slack_var = false, λy = 1.0, λg = 0.0)
Constructs and solves the DeePC optimization problem, with a slack variable or without (noisy/nonlinear or not).
It returns `uₜ` and `yₜ`, the input and output sequence solutions of the DeePC over the N-horizon.
The regularization weights (the λ's) are set to zero and the slack variable is deactivated in the default settings.
"""
function run_DeePC(prob::DeePC_struct, Uᵢₙᵢ::Vector{Float64}, Yᵢₙᵢ::Vector{Float64}; 
                    slack_var = false, λy = 1.0, λg = 1.0, γ = 0.0)

if slack_var == false
    λy = 1.0; λg = 0.0; # Some weights with no effect on optimization
end

p = size(prob.R)[1]
m = size(prob.Q)[1]
uₖ = Variable(p, prob.N)
yₖ = Variable(m, prob.N+1)
g = Variable(size(prob.UpYp)[2])
σ = Variable(size(Yᵢₙᵢ)[1])



# Define objective function:
cost =  λg * norm(g,1) + λy * norm(σ,1) # this part will have an effect if slack variable is true
for k in 1:prob.N
    cost += quadform(uₖ[:,k], prob.R; assume_psd=true) + quadform(yₖ[:,k], prob.Q; assume_psd=true)
end
cost += quadform(yₖ[:,prob.N+1], prob.Q; assume_psd=true)

# Regularization term enforcing data-consistency
U_ini_matrix = reshape(Uᵢₙᵢ, p, :)
Y_ini_matrix = reshape(Yᵢₙᵢ, m, :)

reg = 0

for k in 1:prob.N
    if k - prob.Tᵢₙᵢ < 0
        #Taking input-output values from recent past data
        Ψₖu = [vec(U_ini_matrix[:, end + 1 + k - prob.Tᵢₙᵢ:end]);
                vec(uₖ[:,1:k])]
        Ψₖy = [vec(Y_ini_matrix[:, end + 1 + k - prob.Tᵢₙᵢ:end]);
                vec(yₖ[:,1:k])]
    else
        Ψₖu = vec(uₖ[:,k-prob.Tᵢₙᵢ+1:k])
        Ψₖy = vec(yₖ[:,k-prob.Tᵢₙᵢ+1:k])
    end
    reg += quadform([Ψₖu; Ψₖy] - mean(prob.UpYp, dims = 2), prob.H_data_inv; assume_psd=true)
end

cost += γ * reg

# Define optimization problem
problem = minimize(cost)

# Define dynamic constraints (data-driven representation of the dynamics):
problem.constraints += [uₖ[:,k] == prob.Uf[(k-1)*p+1:k*p,:] * g for k in 1:prob.N]
problem.constraints += [yₖ[:,k] == prob.Yf[(k-1)*m+1:k*m,:] * g for k in 1:prob.N]

# Include the slack variable for some flexibility when facing noise and nonlinearities:
if slack_var == false
    problem.constraints += [prob.UpYp * g == [Uᵢₙᵢ; Yᵢₙᵢ]]
elseif slack_var == true
    problem.constraints += [prob.UpYp * g == [Uᵢₙᵢ; Yᵢₙᵢ + σ]]
end

"""
# Include the input and output constraints (if they exist)
if prob.Constrained == true
    Mats = prob.constraintMatrices
    if !(Mats.Au isa Nothing || Mats.bu isa Nothing)
        problem.constraints += [Mats.Au * uₖ[:,k] <= Mats.bu for k in 1:prob.N]
    end
    if !(Mats.Ay isa Nothing || Mats.by isa Nothing)
        problem.constraints += [Mats.Ay * yₖ[:,k] <= Mats.by for k in 1:prob.N]
    end
end
"""

solve!(problem, SCS.Optimizer; silent_solver = true)

return uₖ.value, yₖ.value
######
end # function run_DeePC

"""
    build_data_matrix(U::Matrix{Float64}, Y::Matrix{Float64}, Tᵢₙᵢ::Int, N::Int)
Takes the data collected `U,Y` through a persistently excited experiment and returns
the matrices `[Up; Yp], Uf, Yf` used in the construction of the DeePC algorithm.
"""
function build_data_matrix(U::Matrix{Float64}, Y::Matrix{Float64}, Tᵢₙᵢ::Int, N::Int)
    Hu = hankellize(U, Tᵢₙᵢ+N)
    Hy = hankellize(Y, Tᵢₙᵢ+N)

    m = size(U)[1]
    p = size(Y)[1]
    T = size(U)[2]

    Up = Hu[1:Tᵢₙᵢ*m,:];      Uf = Hu[Tᵢₙᵢ*m+1:end,:];  
    Yp = Hy[1:Tᵢₙᵢ*p,:];      Yf = Hy[Tᵢₙᵢ*p+1:end,:];
    UpYp = [Up; Yp]
    Huy = UpYp .- mean(UpYp, dims = 2)
    H_data = 1/(T + 1 - Tᵢₙᵢ) * Huy * Huy' + 1e-3 * I
    return UpYp, Uf, Yf, H_data
end

"""
    hankellize(U::Matrix{Float64}, L::Int) 
Constructs the hankel matrix `H`` of a matrix `U`, with `L` number of shifts. `H` here corresponds to `H_L(U)` in the DeePC paper.
"""
function hankellize(U::Matrix{Float64}, L::Int)
    m, T = size(U)

    if T<L
        error("T is less than L, cannot construct a Hankel matrix.")
    end
    H = zeros(m*L, T-L+1)
    for row=1:L
        H[(row-1)*m+1:row*m,:] = U[:,row:T-L+row]
    end
    return H
end

end # module