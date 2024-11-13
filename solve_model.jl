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
    μ₁ :: Float64 = 8.0
    μ₀ :: Float64 = 0.9*8.0
    λ :: Float64 = 0.01
    γ :: Float64 = 0.09
    σ::Float64 = 9.0
    αₚ::Float64 = 0.02
    κₚ::Float64 = 1.0
    αₐ::Float64 = 0.02
    κₐ::Float64 = 0.3
end


function solve_v1(L₁, params::Params; R₁=0.0, return_z = true)
    @unpack r, μ₁, μ₀, λ, γ, σ, αₚ, κₚ, αₐ, κₐ = params

    Wᵦ = 5*L₁
    
    # Define the system of differential equations
    function hjb!(du, u, p, x)
        v = u[1]
        dv = u[2]
        du[1] = dv
        du[2] = (r * v - μ₁ - γ * x * dv) / (0.5 * λ^2 * σ^2)
    end

    function bca!(residual, u, p)
        # Boundary condition at the beginning of the interval: u(a) = L_1
        residual[1] = u[1] - L₁
    end

    function bcb!(residual, u, p)
        # Boundary conditions at the end of the interval
        residual[1] = u[2] + 1 
        residual[2] = (r * u[1] - μ₁ - γ * Wᵦ * u[2]) / (0.5 * λ^2 * σ^2)  # Second derivative u''(b) = 0
    end

    w_span = (R₁, Wᵦ)
    n = 100
    dt = (Wᵦ - R₁) / n

    function initial_guess(p,x)
        v = L₁
        dv = 10.0
        return [v, dv]
    end


    # Solve the BVP
    bvp = TwoPointBVProblem(hjb!, (bca!, bcb!), initial_guess, w_span, bcresid_prototype = (zeros(1), zeros(2)))

    solution = solve(bvp, MIRK4(), dt=dt)

    vs = [solution.u[i][1] for i in 1:length(solution.u)]
    dvs = [solution.u[i][2] for i in 1:length(solution.u)]

    # maximum and argmax of the value function
    v_max = maximum(vs)
    w_max = solution.t[argmax(vs)]

    # interpolate vs 
    v = interpolate((solution.t,), vs, Gridded(Linear()))
    # extrapolate linearly
    v = extrapolate(v, Line())
    # loss function 
    if return_z
        (1-αₚ)*v_max - κₚ - v((1-αₐ)*w_max - κₐ)
    else
        # return R 
        return (1-αₐ)*w_max-κₐ, (1-αₚ)*v_max - κₚ , vs, dvs
    end
    
end


# Numerical gradient using finite differences
function numerical_derivative(f, x; h=1e-8)
    (f(x + h) - f(x)) / h
end

# Numerical second derivative using finite differences
function numerical_second_derivative(f, x; h=1e-8)
    (f(x + h) - 2 * f(x) + f(x - h)) / (h^2)
end

# Newton's method with numerical derivatives
function newtons_method(f; x0=0.0, tolerance=1e-6, max_iters=1000)
    x = x0
    for i in 1:max_iters
        # Calculate numerical first and second derivatives
        grad = numerical_derivative(f, x)

        # Newton's update step
        x_new = x - f(x) / grad

        # Print iteration details
        @printf("Iteration %d: x = %.6f, f(x) = %.6f\n", i, x, f(x))

        # Check for convergence
        if abs(x_new - x) < tolerance
            println("Converged!")
            return x_new
        end

        x = x_new
    end
    println("Reached maximum iterations.")
    return x
end



# Run gradient descent
function main()
    params = Params()
    f = L -> solve_v1(L, params)
    # Run Newton's method
    optimal_x = newtons_method(f, x0=10.0)
    println("Optimal x found: ", optimal_x)

    # solve again with that L and return R
    R, L, vs, dvs = solve_v1(optimal_x, params, R₁=0.0, return_z=false)
    # print the new R 
    println("Optimal R found: ", R)

    # iterate until R converges
    tol = 1e-6
    last_R = copy(R)
    last_L = copy(L)
    for i in 1:1000
        # solve again with that L and return R
        R, L, vs, dvs = solve_v1(last_L, params, R₁=last_R, return_z=false)

        # multiples of 20 to display
        if i % 20 == 0
            println("Iteration: ", i, " R: ", R, " L: ", L)
        end
        
        if (abs(R - last_R) < tol) && (abs(L - last_L) < tol)
            break
        end
        last_R = copy(R)
        last_L = copy(L)
    end

    println("Converged R: ", R)
end

main()