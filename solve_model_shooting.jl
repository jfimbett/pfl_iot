using Plots
using Parameters
using Distributions
using LinearAlgebra
using Random
using ProgressBars
using LaTeXStrings
using Roots
using NLsolve
using Interpolations
using Integrals
using Suppressor
using JumpProcesses
using Optim
using DifferentialEquations
using Printf

@with_kw struct Params
    r::Float64 = 0.04
    μ₁:: Float64 = 8.0
    μ₀:: Float64 = 0.9*8.0
    λ:: Float64 = 0.29
    γ:: Float64 = 0.09
    σ::Float64 = 9.0
    αₚ::Float64 = 0.05
    κₚ::Float64 = 5
    αₐ::Float64 = 0.02
    κₐ::Float64 = 5
end



function solve_v1(L₁; params = Params(), R₁=0.0)
    @unpack r, μ₁, γ, λ, σ, αₚ, κₚ, αₐ, κₐ = params
    Wᵦ = 5*R₁+5.0
    # Define the system of differential equations
    function hjb!(du, u, p, x)
        v = u[1]
        dv = u[2]
        du[1] = dv
        du[2] = (r * v - μ₁ - γ * x * dv) / (0.5 * λ^2 * σ^2)
    end

    function bca!(residual, u, p)
        residual[1] = u[1] - L₁
    end

    function bcb!(residual, u, p)
        # Boundary conditions at the end of the interval
        residual[1] = u[2] + 1 
        residual[2] = (r * u[1] - μ₁ - γ * Wᵦ * u[2]) / (0.5 * λ^2 * σ^2)  # Second derivative u''(b) = 0
    end

    w_span = (R₁, Wᵦ)
    n = 100
    

    function initial_guess(p,x)
        v = L₁
        dv = 10.0
        return [v, dv]
    end


    dt = (Wᵦ - R₁) / n
    # Solve the BVP
    bvp = TwoPointBVProblem(hjb!, (bca!, bcb!), initial_guess, w_span, bcresid_prototype = (zeros(1), zeros(2)))

    solution = solve(bvp, MIRK4(), dt=dt)

    vs = [solution.u[i][1] for i in 1:length(solution.u)]

    # maximum and argmax of the value function
    v_max = maximum(vs)
    w_max = solution.t[argmax(vs)]
    
    # interpolate vs 
    v = interpolate((solution.t,), vs, Gridded(Linear()))
    # extrapolate linearly
    v = extrapolate(v, Line())
    # loss function 

    # plot w vs v
    #plot(solution.t, vs, label = "v(w)", xlabel = "w", ylabel = "v", title = "Value function")
        
    return (1-αₚ)*v_max - κₚ - v((1-αₐ)*w_max - κₐ), solution, v
end

# for zero
z, solution, v = solve_v1(0.0)

# plot
plot(solution.t, [solution.u[i][1] for i in 1:length(solution.u)], label = "v(w)", xlabel = "w", ylabel = "v", title = "Value function")