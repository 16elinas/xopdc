using NPZ
using Interpolations
using Cubature
using PyCall
using LinearAlgebra
@pyimport matplotlib.animation as anim
# using PyPlot
using QuadGK

# using Plots

rcParams = PyDict(matplotlib["rcParams"])
rcParams["font.size"] = 10;
rcParams["font.sans-serif"] = "Arial";
rcParams["axes.labelsize"] = 12;
rcParams["xtick.labelsize"] = 12;
rcParams["ytick.labelsize"] = 12;
rcParams["legend.fontsize"] = 12;
rcParams["text.usetex"] = "False";
rcParams["svg.fonttype"] = "none";
rcParams["contour.negative_linestyle"] = "solid";
rcParams["contour.linewidth"] = 0.75;
rcParams["pcolor.shading"] = "auto";
rcParams["lines.linewidth"] = 0.75;
# rc("font", family="sans-serif")
# rc("font", size=12.0)
# rc("svg", fonttype="none")

const ħ = 1.06e-34
const ħeV = 6.582e-16
const ϵ0 = 8.854e-12
const qe = 1.6e-19;
const c = 2.99792e8

struct PumpBeam
    ωp::Real
    θp::Real
    ϕp::Real
end

struct PhC_Data
    a::Real #periodicity in meters
    Mspline
    ωspline
    vgspline
end

struct PhC_1D_Data
    a::Real #periodicity in meters
    Mspline
end

# unit conversions from MPB to real units
function MPB_to_rad(ω::Float64, D::PhC_Data)
    return ω * c / D.a
end

function MPB_to_eV(ω::Float64, D::PhC_Data)
    return ω * c / D.a * ħeV
end
    
function MPB_to_k(k::Float64, D::PhC_Data)
    return k * 2π / D.a
end

function dΓdωdΩ_free(ωs::AbstractFloat, θs::AbstractFloat, ϕs::AbstractFloat, gvec, P::PumpBeam, ϵbg::Real, χ::AbstractFloat, np::AbstractFloat, L::AbstractFloat; lz::AbstractFloat=0.2e14, pol="all")
    ωp = P.ωp
    θp = P.θp 
    ϕp = P.ϕp
    
    kp = ωp/c
    ks = ωs/c
    
#     kpvec = kp*[sin(θp)*cos(ϕp), sin(θp)*sin(ϕp), cos(θp)]
#     ksvec = ks*[sin(θs)*cos(ϕs), sin(θs)*sin(ϕs), cos(θs)]
    

#     k0vec = gvec .+ kpvec .- ksvec
    g = gvec[1] #assuming in the x direction
    

    kiz = -ks*sin(θs) - kp*sin(θp) + gvec[3]
    kiy = -ks*cos(θs)*sin(ϕs) + kp*cos(θp)*sin(ϕp) + gvec[2]
    kix = - ks*cos(θs)*cos(ϕs) + kp*cos(θp)*cos(ϕp) + gvec[1]

    kivec = [kix, kiy, kiz]
    
    ki = norm(kivec)
    ωi = c*ki/sqrt(ϵbg) #dispersion in the medium

    
    Ep2 = ħ*np*ωp/(2*ϵ0)
    prefactor =  ωs^3/(16*π^2*c^3*ϵbg) * Ep2 * L^3 * χ^2
#     idler_pol = 1 - (sum(kivec.*normalize(gvec))/ki)^2
    if pol == "te"
        idler_pol = abs(sum(normalize(kivec).*normalize(gvec)))^2
    elseif pol == "tm"
        idler_pol = 1 - abs(sum(normalize(kivec).*normalize(gvec)))^2
    else 
        idler_pol = 1
    end 

    return prefactor * (ωp-ωs) * lor(ωp - ωs - ωi, lz) *idler_pol
end

function info_free(ωs, θs, ϕs, gvec, P::PumpBeam, ϵbg, χ, np, L; lz=0.2e14, pol="te")
    ωp = P.ωp
    θp = P.θp 
    ϕp = P.ϕp
    
    kp = ωp/c
    ks = ωs/c
    
#     kpvec = kp*[sin(θp)*cos(ϕp), sin(θp)*sin(ϕp), cos(θp)]
#     ksvec = ks*[sin(θs)*cos(ϕs), sin(θs)*sin(ϕs), cos(θs)]
    

#     k0vec = gvec .+ kpvec .- ksvec
    g = gvec[1] #assuming in the x direction
    

    kiz = -ks*sin(θs) - kp*sin(θp) + gvec[3]
    kiy = -ks*cos(θs)*sin(ϕs) + kp*cos(θp)*sin(ϕp) + gvec[2]
    kix = - ks*cos(θs)*cos(ϕs) + kp*cos(θp)*cos(ϕp) + gvec[1]

    kivec = [kix, kiy, kiz]
    
    ki = norm(kivec)
    ωi = c*ki/sqrt(ϵbg) #dispersion in the medium
    
    kpx = -kp*sin(θp)
    kpy = kp*cos(θp)*sin(ϕp)
    kpz = kp*cos(θp)*cos(ϕp)
    kpvec = [kpx, kpy, kpz]
    
    Ep2 = ħ*np*ωp/(2*ϵ0)
    prefactor =  ωs^3/(16*π^2*c^3*ϵbg) * Ep2 * L^3 * χ^2
#     idler_pol = 1 - (sum(kivec.*normalize(gvec))/ki)^2
    if pol == "te"
        idler_pol = abs(sum(normalize(kivec).*normalize(gvec)))^2
    else
        idler_pol = 1 - abs(sum(normalize(kivec).*normalize(gvec)))^2
    end
    

    return prefactor, ωi, lor(ωp - ωs - ωi, lz), idler_pol, (sum(kivec.*normalize(gvec))/ki)^2, kivec,
     (ωp-ωs)
end

function lor(x, η)
    return 1/π * η / (x^2 + η^2)
end


function dΓdωdΩ_1D(ωs::AbstractFloat, θs::AbstractFloat, ϕs::AbstractFloat, gvec, P::PumpBeam, D::PhC_Data, χ::AbstractFloat, np::AbstractFloat, L::AbstractFloat, band::Integer, ϵbg::Real=1; lz::AbstractFloat=0.2e14, pol="te")
    ωp = P.ωp
    θp = P.θp 
    ϕp = P.ϕp
    
    kp = ωp/c
    ks = ωs/c
    
    a = D.a

    kiz = -ks*sin(θs) - kp*sin(θp) + gvec[3]
    kiy = -ks*cos(θs)*sin(ϕs) + kp*cos(θp)*sin(ϕp) + gvec[2]
    kB = - ks*cos(θs)*cos(ϕs) + kp*cos(θp)*cos(ϕp) + gvec[1]
    kivec = [kB, kiy, kiz]
   
    ki = norm(kivec)
    
    k0z = kiz * a/(2π)
    k0y = kiy * a/(2π)

    k0x = (kB * a / (2π) + 100.5) % 1 - 0.5 # in MPB units
    k0par = sqrt(k0z^2 + k0y^2)
    
    if k0par > 4
        return 0
    end
    
    noffset = -floor(kB*a/(2π) + 0.5)
    
    # We only have mode overlap data for the first 3 offsets in all directions
    if abs(noffset) > 3
        return 0
    end
    nz = -noffset + 4
#     println(nx, ny, nz)

    Msplines = [D.Mspline[band, Integer(nz),i] for i in 1:2]
    ωspline = D.ωspline[band]

    ω0 = ωspline(k0x, k0par)
#     M0 = sum([Msplines[i](k0z, k0y, -k0x) for i in 1:3].* normalize(gvec))
#     M0 = M0[1]
    M02 = sum([Msplines[i](k0x, k0par)^2 for i in 1:2])

#     println("M0: ", M0)

    ωi = ω0 * 2π*c / (a)#*sqrt(ϵbg))
    Ep2 = ħ*np*ωp/(2*ϵ0)
    prefactor = ωs^3/(16*π^2*c^3*ϵbg) * Ep2 * L^3 * χ^2
#     idler_pol = 1 - (sum(kivec.*normalize(gvec))/ki)^2
    if pol == "te"
        polfactor = abs(sum(normalize(kivec).*normalize(gvec)))^2
    else
        polfactor = 1 - abs(sum(normalize(kivec).*normalize(gvec)))^2
    end
    
    return prefactor * (ωp-ωs) * M02 * lor(ωp - ωs - ωi, lz) * polfactor
end


function info_1D(ωs::AbstractFloat, θs::AbstractFloat, ϕs::AbstractFloat, gvec, P::PumpBeam, D::PhC_Data, χ::AbstractFloat, np::AbstractFloat, L::AbstractFloat, band::Integer, ϵbg::Real=1; lz::AbstractFloat=0.2e14, pol="te")
    ωp = P.ωp
    θp = P.θp 
    ϕp = P.ϕp
    
    kp = ωp/c
    ks = ωs/c
    
    a = D.a

    kiz = -ks*sin(θs) - kp*sin(θp) + gvec[3]
    kiy = -ks*cos(θs)*sin(ϕs) + kp*cos(θp)*sin(ϕp) + gvec[2]
    kB = - ks*cos(θs)*cos(ϕs) + kp*cos(θp)*cos(ϕp) + gvec[1]
    kivec = [kB, kiy, kiz]
   
    ki = norm(kivec)
    
    k0z = kiz * a/(2π)
    k0y = kiy * a/(2π)

    k0x = (kB * a / (2π) + 100.5) % 1 - 0.5 # in MPB units
    k0par = sqrt(k0z^2 + k0y^2)
    
    if k0par > 4
        return 0,0,0,0,0,0, [0,0,0], 0, 0, 0, 0, 0, 0
    end
    
    noffset = -floor(kB*a/(2π) + 0.5)
    
    # We only have mode overlap data for the first 3 offsets in all directions
    if abs(noffset) > 3
        return 0,0,0,0,0,0, [0,0,0], 0, 0, 0, 0, 0, 0
    end
    nz = -noffset + 4
#     println(nx, ny, nz)

    Msplines = [D.Mspline[band, Integer(nz),i] for i in 1:2]
    ωspline = D.ωspline[band]

    ω0 = ωspline(k0x, k0par)
#     M0 = sum([Msplines[i](k0z, k0y, -k0x) for i in 1:3].* normalize(gvec))
#     M0 = M0[1]
      M02 = sum([Msplines[i](k0x, k0par)^2 for i in 1:2])


    ωi = ω0 * 2π*c / (a)#*sqrt(ϵbg))
    Ep2 = ħ*np*ωp/(2*ϵ0)
    prefactor = ωs^3/(16*π^2*c^3*ϵbg) * Ep2 * L^3 * χ^2
    if pol == "te"
        polfactor = abs(sum(normalize(kivec).*normalize(gvec)))^2
    else
        polfactor = 1 - abs(sum(normalize(kivec).*normalize(gvec)))^2
    end

    
    return ωi, M02, lor(ωp - ωs - ωi, lz), kB, k0x, noffset, kivec, polfactor, k0y, kiz, k0par, prefactor, (ωp-ωs)
end


function dΓdωdΩ_1D_analytical_1(ωs::AbstractFloat, θs::AbstractFloat, ϕs::AbstractFloat, gvec, P::PumpBeam, D::PhC_Data, χ::AbstractFloat, np::AbstractFloat, L::AbstractFloat, band::Integer, ϵbg::Real=1; lz::AbstractFloat=0.2e14, pol="te")
    ωp = P.ωp
    θp = P.θp 
    ϕp = P.ϕp
    
    kp = ωp/c
    ks = ωs/c
    
    a = D.a
    ϵ1 = 5.3
    ϵ2 = 1

    kiz = -ks*sin(θs) - kp*sin(θp) + gvec[3]
    kiy = -ks*cos(θs)*sin(ϕs) + kp*cos(θp)*sin(ϕp) + gvec[2]
    kB = - ks*cos(θs)*cos(ϕs) + kp*cos(θp)*cos(ϕp) + gvec[1]
    k0z = kiz * a/(2π)
    k0y = kiy * a/(2π)
    k0x = (kB * a / (2π) + 100.5) % 1 - 0.5 # in MPB units
    
    noffset = -floor(kB*a/(2π) + 0.5) # g
    kpar = sqrt(kiz^2 + kiy^2)
    k0par = kpar *a/(2π)

    
    if abs(k0par) > 4 # we don't have enough data for that
        return 0
    end

    ωspline = D.ωspline[band]
    ω0 = ωspline(k0x, k0par)
    ωi = ω0 * 2π*c / (a*sqrt(ϵbg)) # ϵbg to be used for free space simulations; 1 otherwise
    
    if ϵ1*ωi^2/c^2 - kpar^2 < 0 && ωi^2/c^2 - kpar^2 < 0
        return 0
    end
    
    # do the mode integral analytically
    # but first get the mode coefficients numerically...
    coeffs, norm = mode_coeffs_and_norm(ωi, kpar, ϵ1, ϵ2, a/2, a, pol)
    a1 = 1/norm
    b1 = coeffs[1]/norm
    a2 = coeffs[2]/norm
    b2 = coeffs[3]/norm
    
    k1 = sqrt(Complex(5.3*ωi^2/c^2 - kpar^2))
    k2 = sqrt(Complex(ωi^2/c^2 - kpar^2))
    
    # Julia uses the normalized sinc function
    g = -2π*noffset/a
    
    zs = range(0, 31*a/64, length=32)
    M2 = abs(quadgk(z -> efield_fast(z, ωi, kpar, a, 5.3, a1, b1, a2, b2, k1, k2, k0x*2π/a, pol=pol)*exp(2im*π*noffset/a*z), 0, a/2)[1])^2/a
    
    Ep2 = ħ*np*ωp/(2*ϵ0)
    prefactor = ωs^3/(16*π^2*c^3*ϵbg) * Ep2 * L^3 * χ^2 #* a/2
    if pol == "te"
        polfactor = abs(sum(normalize([kB, kiy, kiz]).*normalize(gvec)))^2
    else
        polfactor = 1 - abs(sum(normalize([kB, kiy, kiz]).*normalize(gvec)))^2
    end
    dΓdωdΩ = prefactor * ωi * M2 * lor(ωp - ωs - ωi, lz) * polfactor 
    return dΓdωdΩ
end


function in_BZ(k)
    for elt in k
        if elt > 0.5 || elt < -0.5
            return false
        end
    end
    return true
end

function dΓdωdΩ_SC(ωs::AbstractFloat, θs::AbstractFloat, ϕs::AbstractFloat, gvec, P::PumpBeam, D::PhC_Data, χ::AbstractFloat, np::AbstractFloat, L::AbstractFloat, band::Integer, ϵbg::Real=1; lz::AbstractFloat=0.2e14)
    ωp = P.ωp
    θp = P.θp 
    ϕp = P.ϕp
    
    kp = ωp/c
    ks = ωs/c
    
    a = D.a

    kBx = -ks*sin(θs) - kp*sin(θp) + gvec[1]
    kBy = -ks*cos(θs)*sin(ϕs) + kp*cos(θp)*sin(ϕp) + gvec[2]
    kBz = - ks*cos(θs)*cos(ϕs) + kp*cos(θp)*cos(ϕp) + gvec[3]

    kBvec = [kBx, kBy, kBz]# .+ Goffset # in meters

#     k0_vec_no_unit = kBvec * a / (2π)

    k0_vec_no_unit = (kBvec * a / (2π) .+ 100.5) .% 1 .- 0.5 # in MPB units
    if !in_BZ(k0_vec_no_unit)
#         println("why are we here")
        return 0
    end
    
    noffset = floor.(kBvec*a/(2π) .+ 0.5)
    
    # We only have mode overlap data for the first 3 offsets in all directions
    if any(abs.(noffset) .> 3)
        return 0
    end
    nx, ny, nz = noffset .+ [4,4,4]
#     println(nx, ny, nz)

    Msplines = [D.Mspline[band, Integer(nx), Integer(ny), Integer(nz),i] for i in 1:3]

    ωspline = D.ωspline[band]
    
    k0x, k0y, k0z = k0_vec_no_unit

    ω0 = ωspline(k0x, k0y, k0z)
#     M0 = Mspline(k0x, k0y, k0z)
    M02 = sum([Msplines[i](k0x, k0y, k0z) for i in 1:3].^2)
#     M0 = M0[1]
#     println("M0: ", M0)

    ωi = ω0 * 2π*c / (a*sqrt(ϵbg))
    Ep2 = ħ*np*ωp/(2*ϵ0)
    prefactor = ωs^3/(16*π^2*c^3*ϵbg) * Ep2 * L^3 * χ^2
    
    return prefactor * ωi * M02 * lor(ωp - ωs - ωi, lz)
end



function bin(x, divs) #assume that divs is ordered
    for (i, elt) in enumerate(divs)
        if x >= elt && x < divs[i+1]
            return i
        end
    end
end


function info_SC(ωs, θs, ϕs, gvec, P::PumpBeam, D::PhC_Data, χ, np, L, band, ϵbg=1; lz=0.2e14)
    ωp = P.ωp
    θp = P.θp 
    ϕp = P.ϕp
    
    kp = ωp/c
    ks = ωs/c
    
    a = D.a

    kBx = -ks*sin(θs) - kp*sin(θp) + gvec[1]
    kBy = -ks*cos(θs)*sin(ϕs) + kp*cos(θp)*sin(ϕp) + gvec[2]
    kBz = - ks*cos(θs)*cos(ϕs) + kp*cos(θp)*cos(ϕp) + gvec[3]

    kBvec = [kBx, kBy, kBz]# .+ Goffset # in meters

    k0_vec_no_unit = (kBvec * a / (2π) .+ 100.5) .% 1 .- 0.5 # in MPB units
#     if !in_BZ(k0_vec_no_unit)
#         println("why are we here")
#         return [0,0,0], [0,0,0], 0, 0, 0, 0, 0, [0,0,0]
#     end
    
    noffset = floor.(kBvec*a/(2π) .+ 0.5)
    
    # We only have mode overlap data for the first 3 offsets in all directions
    if any(abs.(noffset) .> 3)
        return noffset, k0_vec_no_unit, 0, 0, 0, 0, 0, [0,0,0], [0,0,0]
    end
    nx, ny, nz = noffset .+ [4,4,4]

    Msplines = [D.Mspline[band, Integer(nx), Integer(ny), Integer(nz),i] for i in 1:3]

    ωspline = D.ωspline[band]
    
    k0x, k0y, k0z = k0_vec_no_unit

    ω0 = ωspline(k0x, k0y, k0z)
    M02 = sum(([Msplines[i](k0x, k0y, k0z) for i in 1:3].* normalize(gvec)).^2)

#     println("M0: ", M0)

    ωi = ω0 * 2π*c / (a*sqrt(ϵbg))
    Ep2 = ħ*np*ωp/(2*ϵ0)
    prefactor = ωs^3/(16*π^2*c^3*ϵbg) * Ep2 * L^3 * χ^2

    return noffset, k0_vec_no_unit, M02, ωi, lor(ωp - ωs - ωi, lz), Ep2, prefactor, kBvec, [Msplines[i](k0x, k0y, k0z) for i in 1:3]
end


# find the noffset for the simple cubic lattice
function noffset_SC(ωs, θs, ϕs, gvec, P::PumpBeam, D::PhC_Data; a=0.5e-6)
    ωp = P.ωp
    θp = P.θp 
    ϕp = P.ϕp
    
    kp = ωp/c
    ks = ωs/c

    a = D.a

    kBx = -ks*sin(θs) - kp*sin(θp) + gvec[1]
    kBy = -ks*cos(θs)*sin(ϕs) + kp*cos(θp)*sin(ϕp) + gvec[2]
    kBz = - ks*cos(θs)*cos(ϕs) + kp*cos(θp)*cos(ϕp) + gvec[3]

    kBvec = [kBx, kBy, kBz]# .+ Goffset # in meters

    k0_vec_no_unit = (kBvec * a / (2π) .+ 100.5) .% 1 .- 0.5 # in MPB units
#     if !in_BZ(k0_vec_no_unit)
#         println("why are we here")
#         return [0,0,0], [0,0,0], 0, 0, 0, 0, 0, [0,0,0]
#     end
    
    return -floor.(kBvec*a/(2π) .+ 0.5)
end


## This function does not dot product the M matrix with the G vector
function dΓdωdΩ_SC_fs(ωs::AbstractFloat, θs::AbstractFloat, ϕs::AbstractFloat, gvec, P::PumpBeam, D::PhC_Data, χ::AbstractFloat, np::AbstractFloat, L::AbstractFloat, band::Integer, ϵbg::Real=1; lz::AbstractFloat=0.2e14, pol="te", a::AbstractFloat=0.8e-7)
    ωp = P.ωp
    θp = P.θp 
    ϕp = P.ϕp
    
    kp = ωp/c
    ks = ωs/c
    
    a = D.a

    kBx = -ks*sin(θs) - kp*sin(θp) + gvec[1]
    kBy = -ks*cos(θs)*sin(ϕs) + kp*cos(θp)*sin(ϕp) + gvec[2]
    kBz = - ks*cos(θs)*cos(ϕs) + kp*cos(θp)*cos(ϕp) + gvec[3]

    kBvec = [kBx, kBy, kBz]# .+ Goffset # in meters

#     k0_vec_no_unit = kBvec * a / (2π)

    k0_vec_no_unit = (kBvec * a / (2π) .+ 100.5) .% 1 .- 0.5 # in MPB units
    if !in_BZ(k0_vec_no_unit)
#         println("why are we here")
        return 0
    end
    
    noffset = -floor.(kBvec*a/(2π) .+ 0.5)
    
    # We only have mode overlap data for the first 3 offsets in all directions
    if any(abs.(noffset) .> 3)
        return 0
    end
    nx, ny, nz = -noffset .+ [4,4,4]
#     println(nx, ny, nz)

    Msplines = [D.Mspline[band, Integer(nx), Integer(ny), Integer(nz),i] for i in 1:3]

    ωspline = D.ωspline[band]
    
    k0x, k0y, k0z = k0_vec_no_unit

    ω0 = ωspline(k0x, k0y, k0z)
#     M0 = Mspline(k0x, k0y, k0z)
    M0 = sum(([Msplines[i](k0x, k0y, k0z) for i in 1:3].* normalize([1, 1, 1])).^2)
#     M0 = M0[1]
#     println("M0: ", M0)

    ωi = ω0 * 2π*c / (a*sqrt(ϵbg))
    Ep2 = ħ*np*ωp/(2*ϵ0)
    prefactor = ωs^3/(16*π^2*c^3*ϵbg) * Ep2 * L^3 * χ^2
    
    return prefactor * ωi * M0 * lor(ωp - ωs - ωi, lz)
end

function dΓdωdΩ_SC_fs_nn(ωs::AbstractFloat, θs::AbstractFloat, ϕs::AbstractFloat, gvec, P::PumpBeam, D::PhC_Data, χ::AbstractFloat, np::AbstractFloat, L::AbstractFloat, band::Integer, ϵbg::Real=1; lz::AbstractFloat=0.2e14, Nk::Integer=15)
    ωp = P.ωp
    θp = P.θp 
    ϕp = P.ϕp
    
    kp = ωp/c
    ks = ωs/c
    
    a = D.a

    kBx = -ks*sin(θs) - kp*sin(θp) + gvec[1]
    kBy = -ks*cos(θs)*sin(ϕs) + kp*cos(θp)*sin(ϕp) + gvec[2]
    kBz = - ks*cos(θs)*cos(ϕs) + kp*cos(θp)*cos(ϕp) + gvec[3]

    kBvec = [kBx, kBy, kBz]# .+ Goffset # in meters

    k0_vec_no_unit = (kBvec * a / (2π) .+ 100.5) .% 1 .- 0.5 # in MPB units
    if !in_BZ(k0_vec_no_unit)
#         println("why are we here")
        return 0
    end
    
    noffset = -floor.(kBvec*a/(2π) .+ 0.5)
    
    # We only have mode overlap data for the first 3 offsets in all directions
    if any(abs.(noffset) .> 3)
        return 0
    end
    nx, ny, nz = -noffset .+ [4,4,4]
#     println(nx, ny, nz)

    Msplines = [D.Mspline[band, Integer(nx), Integer(ny), Integer(nz),i] for i in 1:3]

    ωspline = D.ωspline[band]
    
    k0x, k0y, k0z = k0_vec_no_unit

    ω0 = ωspline(k0x, k0y, k0z)
    ik = round.(k0_vec_no_unit.*(Nk-1) .+ ((Nk+1)/2))
    k0_rounded = (ik.-((Nk+1)/2)) ./ (Nk-1)
    k0x, k0y, k0z = k0_rounded

#     M0 = Mspline(k0x, k0y, k0z)
    M0 = sum(([Msplines[i](k0x, k0y, k0z) for i in 1:3].* normalize([1, 1, 1])).^2)
#     M0 = M0[1]
#     M0 = 1/12 
#     println("M0: ", M0)

    ωi = ω0 * 2π*c / (a*sqrt(ϵbg))
    Ep2 = ħ*np*ωp/(2*ϵ0)
    prefactor = ωs^3/(16*π^2*c^3*ϵbg) * Ep2 * L^3 * χ^2
    
    return prefactor * ωi * M0 * lor(ωp - ωs - ωi, lz)
end

# multiply a Cartesian coordinate vector by this matrix to
# take it to reciprocal lattice basis
cartesian_to_BCC_recip = 1/2 * [1 1 -1; -1 1 1; 1 -1 1]
BCC_to_cartesian_recip = [1 0 1; 1 1 0; 0 1 1]
BCC_recip_basis = [[1 1 0], [0 1 1], [1 0 1]]
BCC_basis = [[1/2 1/2 -1/2], [-1/2 1/2 1/2], [1/2 -1/2 1/2]]


# input vector elements are in Cartesian MPB unit basis
function in_1BZ_BCC(k)::Bool
    k = map(x -> round(x, digits=4), reshape(k, 1,3))
    # primitive reciprocal lattice vectors
    b1 = [1 1 0]
    b2 = [0 1 1]
    b3 = [1 0 1]
    bs = [b1,b2,b3]
    # check linear combos of 1 of b1,b2,b3
    for i in 1:3
        for sign in -1:2:1
            if ((k-0.5*sign*bs[i])*transpose(normalize(sign*bs[i])))[1] > 0
                return false
            end
        end
    end
#     signs = [[1,-1],[-1,1]]
    # check combos of 2 of them, always with opposite signs
    for i in 2:3
        for j in 1:2
            if i != j
                for sign in -1:2:1
                    a = 0.5*sign*bs[i]-0.5*sign*bs[j]
                    if ((k-a)*transpose(normalize(a)))[1] > 0
                        return false
                    end
                end
            end
        end
    end
#     k_BCC = cartesian_to_BCC_recip * transpose(k)
#     if abs(k_BCC[1]) > 1/2 || abs(k_BCC[2]) > 1/2 || abs(k_BCC[3]) > 1/2
#         return false
#     end
    return true
                
end

# check if a position is in the wigner-seitz cell of the BCC lattice
function in_ws_cell_BCC(r)
#     for i in 1:3
#         for sign in -1:2:1
#             if ((r-0.5*sign*BCC_basis[i])*transpose(normalize(sign*BCC_basis[i])))[1] > 0
#                 return false
#             end
#         end
#     end
    sign_options = collect(Base.Iterators.product([-1,0,1],[-1,0,1],[-1,0,1]))
    for s in sign_options
        if ((r-0.5*sum(s.*BCC_basis))*transpose(normalize(sum(s.*BCC_basis))))[1] > 0
            return false
        end
    end
    return true
end

# returns k and offsets in the BCC recip lattice vector basis
function bring_into_1BZ_BCC(kB0x::AbstractFloat, kB0y::AbstractFloat, kB0z::AbstractFloat)
    kcart = [kB0x, kB0y, kB0z]
    offset = [0, 0, 0] # in the BCC reciprocal lattice basis
    k_r = cartesian_to_BCC_recip * kcart
    if !in_1BZ_BCC(kcart)
        if abs(k_r[1]) > 1/2
            offset[1] = -round(k_r[1])
            k_r[1] += offset[1]
#             kcart  = kcart .+ offset[1] .* transpose(BCC_recip_basis[1])
        end
       if abs(k_r[2]) > 1/2
            offset[2] = -round(k_r[2])
            k_r[2] += offset[2]
#             kcart  = kcart .+ (offset[2] .* transpose(BCC_recip_basis[2]))
        end
        if abs(k_r[3]) > 1/2
            offset[3] = -round(k_r[3])
            k_r[3] += offset[3]
#             kcart = kcart .+ offset[3] .* transpose(BCC_recip_basis[3])
        end 
    end
    sign_options = collect(Base.Iterators.product([-1,0,1],[-1,0,1],[-1,0,1]))
    if !in_1BZ_BCC(BCC_to_cartesian_recip * k_r)
        for s in sign_options
            if in_1BZ_BCC(BCC_to_cartesian_recip * (k_r .+ s))
                offset .+= s
                k_r .+= s
                return k_r, offset
            end
        end
        println("None of these options worked :'(")
#         println("The original cartesian coords: $kcart")
#         println("After reducing to below 1/2 mag: $k_r")
#         println("Here are the options:")
#         for s in sign_options
#             println("$(k_r .+ s)")
#         end
    end
    return k_r, offset
end

function dΓdωdΩ_stitch_BCC(ωs::AbstractFloat, θs::AbstractFloat, ϕs::AbstractFloat, gvec, P::PumpBeam, D::PhC_Data, χ::AbstractFloat, np::AbstractFloat, L::AbstractFloat, band::Integer, ϵbg::Real=1; lz::AbstractFloat=0.2e14)
    ωp = P.ωp
    θp = P.θp 
    ϕp = P.ϕp
    
    kp = ωp/c
    ks = ωs/c
    
    a = D.a
#     g = gvec[1] #assuming in the x direction
    
    kBx = -ks*sin(θs) - kp*sin(θp) + gvec[1]
    kBy = -ks*cos(θs)*sin(ϕs) + kp*cos(θp)*sin(ϕp) + gvec[2]
    kBz = -ks*cos(θs)*cos(ϕs) + kp*cos(θp)*cos(ϕp) + gvec[3]
    
    kB0x = kBx * a / (2π)
    kB0y = kBy * a / (2π)
    kB0z = kBz * a / (2π)
    
    # k_r and offset both in recip lattice basis
    k_r, offset = bring_into_1BZ_BCC(kB0x, kB0y, kB0z)
#     k_r0 = cartesian_to_BCC_recip * [kB0x, kB0y, kB0z]
#     k_r = collect(ms.first_brillouin_zone(mp.Vector3(k_r0[1], k_r0[2], k_r0[3])))
#     offset = k_r .- k_r0
    if abs(k_r[1]) > 0.75 || abs(k_r[2]) > 0.75 || abs(k_r[3]) > 0.75
        println("k in BCC basis is > 0.75")
        println(k_r)
        println(BCC_to_cartesian_recip * k_r)
        println(in_1BZ_BCC(BCC_to_cartesian_recip * k_r))
        return 0
    end
    
#     BCC_basis_k = cartesian_to_BCC_recip * reshape(k0, 3, 1)
    
    if abs(offset[1]) > 3 || abs(offset[2]) > 3 || abs(offset[3]) > 3
#         println("offset is too much")
#         println(offset)
        return 0
    end
    
    n1, n2, n3 = offset .+ [4,4,4]
    Msplines = [D.Mspline[band, Integer(n1), Integer(n2), Integer(n3),i] for i in 1:3]
    ωspline = D.ωspline[band]

    kcart = BCC_to_cartesian_recip * k_r
    ω0 = ωspline(k_r[1], k_r[2], k_r[3])
#     M0 = D.Mspline[band, Integer(n1), Integer(n2), Integer(n3), 1](k_r[1], k_r[2], k_r[3])

    M0 = reshape([Msplines[i](k_r[1], k_r[2], k_r[3]) for i in 1:3], 1, 3) * reshape(normalize(gvec), 3, 1)
    M0 = M0[1]

    ωi = ω0 * 2π*c / (sqrt(ϵbg)*a)

    #return func
    
    Ep2 = ħ*np*ωp/(2*ϵ0)
    prefactor = ωs^3/(16*π^2*c^3*ϵbg) * Ep2 * L^3 * χ^2
    
    
    return prefactor * ωi * M0^2 * lor(ωp - ωs - ωi, 0.2e14)
    
        
end

function info_stitch_BCC(ωs, θs, ϕs, gvec, P::PumpBeam, D::PhC_Data, χ, np, L, band, ϵbg=1)
    ωp = P.ωp
    θp = P.θp 
    ϕp = P.ϕp
    
    kp = ωp/c
    ks = ωs/c
    
    a = D.a
#     g = gvec[1] #assuming in the x direction
    
    kBx = -ks*sin(θs) - kp*sin(θp) + gvec[1]
    kBy = -ks*cos(θs)*sin(ϕs) + kp*cos(θp)*sin(ϕp) + gvec[2]
    kBz = -ks*cos(θs)*cos(ϕs) + kp*cos(θp)*cos(ϕp) + gvec[3]
    
    kB0x = kBx * a / (2π)
    kB0y = kBy * a / (2π)
    kB0z = kBz * a / (2π)
    
    # k_r and offset both in recip lattice basis
    k_r, offset = bring_into_1BZ_BCC(kB0x, kB0y, kB0z)
#     k_r0 = cartesian_to_BCC_recip * [kB0x, kB0y, kB0z]
#     k_r = collect(ms.first_brillouin_zone(mp.Vector3(k_r0[1], k_r0[2], k_r0[3])))
#     offset = k_r .- k_r0
    if abs(k_r[1]) > 0.75 || abs(k_r[2]) > 0.75 || abs(k_r[3]) > 0.75
        println("k in BCC basis is > 0.75")
        println(k_r)
        println(BCC_to_cartesian_recip * k_r)
        println(in_1BZ_BCC(BCC_to_cartesian_recip * k_r))
        return [0,0,0],[0,0,0],[0,0,0],0,0,0
    end
    
#     BCC_basis_k = cartesian_to_BCC_recip * reshape(k0, 3, 1)
    
    if abs(offset[1]) > 3 || abs(offset[2]) > 3 || abs(offset[3]) > 3
#         println("offset is too much")
#         println(offset)
        return [0,0,0],[0,0,0],[0,0,0],0,0,0
    end
    
    n1, n2, n3 = offset .+ [4,4,4]
    g_bcc = cartesian_to_BCC_recip * [1,0,0]
#     println(g_bcc)
    Msplines = [D.Mspline[band, Integer(n1), Integer(n2), Integer(n3),i] for i in 1:3]
#     println(Msplines)
    ωspline = D.ωspline[band]

    kcart = BCC_to_cartesian_recip * k_r
    ω0 = ωspline(k_r[1], k_r[2], k_r[3])
#     M0 = D.Mspline[band, Integer(n1), Integer(n2), Integer(n3), 1](k_r[1], k_r[2], k_r[3])

    M0 = reshape([Msplines[i](k_r[1], k_r[2], k_r[3]) for i in 1:3], 1, 3) * reshape(normalize(gvec), 3, 1)
    M0 = M0[1]

    ωi = ω0 * 2π*c / (sqrt(ϵbg)*a)

    #return func
    
    Ep2 = ħ*np*ωp/(2*ϵ0)
    prefactor = ωs^3/(16*π^2*c^3*ϵbg) * Ep2 * L^3 * χ^2
    
    return offset, kcart, k_r, M0, ωi, lor(ωp - ωs - ωi, 0.2e14)   
end

FCC_to_cartesian_recip = [1 1 -1; -1 1 1; 1 -1 1]
cartesian_to_FCC_recip = 1/2 * [1 0 1; 1 1 0; 0 1 1]
FCC_recip_basis = [[1 -1 1], [1 1 -1], [-1 1 1]]
FCC_basis = [[1/2 0 1/2], [1/2 1/2 0], [0 1/2 1/2]]
sign_options = collect(Base.Iterators.product([-1,0,1],[-1,0,1],[-1,0,1]))
vec_options = [sum(s.*FCC_recip_basis) for s in sign_options]

function in_1BZ_FCC(k)::Bool
    k = reshape(k, 1,3)
    for v in vec_options
        if ((k-0.5*v)*transpose(normalize(v)))[1] > 0
            return false
        end
    end
    return true
end

function in_1BZ_FCC_2(k)
    for v in vec_options
        if v != [0 0 0]
            if norm(k .- v) < norm(k)
                return false
            end
        end
    end
    return true
end

# returns k and offsets in the BCC recip lattice vector basis
function bring_into_1BZ_FCC(kB0x::AbstractFloat, kB0y::AbstractFloat, kB0z::AbstractFloat)
    kcart = [kB0x, kB0y, kB0z]
    offset = [0, 0, 0] # in the BCC reciprocal lattice basis
    k_r = cartesian_to_FCC_recip * kcart
    if !in_1BZ_FCC_2(transpose(kcart))
        if abs(k_r[1]) > 1
            offset[1] = -round(k_r[1])
            k_r[1] += offset[1]
#             kcart  = kcart .+ offset[1] .* transpose(BCC_recip_basis[1])
        end
       if abs(k_r[2]) > 1
            offset[2] = -round(k_r[2])
            k_r[2] += offset[2]
#             kcart  = kcart .+ (offset[2] .* transpose(BCC_recip_basis[2]))
        end
        if abs(k_r[3]) > 1
            offset[3] = -round(k_r[3])
            k_r[3] += offset[3]
#             kcart = kcart .+ offset[3] .* transpose(BCC_recip_basis[3])
        end 
    end
    if !in_1BZ_FCC(transpose(FCC_to_cartesian_recip * k_r))
        for s in sign_options
            if in_1BZ_FCC(transpose(FCC_to_cartesian_recip * (k_r .+ s)))
                offset .+= s
                k_r .+= s
                return k_r, offset
            end
        end
        println("None of these options worked :'(")
#         println("The original cartesian coords: $kcart")
#         println("After reducing to below 1/2 mag: $k_r")
#         println("Here are the options:")
#         for s in sign_options
#             println("$(k_r .+ s)")
#         end
    end
    return k_r, offset
end

function dΓdωdΩ_stitch_FCC(ωs::AbstractFloat, θs::AbstractFloat, ϕs::AbstractFloat, gvec, P::PumpBeam, D::PhC_Data, χ::AbstractFloat, np::AbstractFloat, L::AbstractFloat, band::Integer, ϵbg::Real=1; lz::AbstractFloat=0.2e14)
    ωp = P.ωp
    θp = P.θp 
    ϕp = P.ϕp
    
    kp = ωp/c
    ks = ωs/c
    
    a = D.a
    
    kBx = -ks*sin(θs) - kp*sin(θp) + gvec[1]
    kBy = -ks*cos(θs)*sin(ϕs) + kp*cos(θp)*sin(ϕp) + gvec[2]
    kBz = -ks*cos(θs)*cos(ϕs) + kp*cos(θp)*cos(ϕp) + gvec[3]
    
    kB0x = kBx * a / (2π)
    kB0y = kBy * a / (2π)
    kB0z = kBz * a / (2π)
    
    
    # k_r and offset both in recip lattice basis
    k_r, offset = bring_into_1BZ_FCC(kB0x, kB0y, kB0z)
#     k_r0 = cartesian_to_BCC_recip * [kB0x, kB0y, kB0z]
#     k_r = collect(ms.first_brillouin_zone(mp.Vector3(k_r0[1], k_r0[2], k_r0[3])))
#     offset = k_r .- k_r0
    kcart = FCC_to_cartesian_recip * k_r
    if abs(k_r[1]) > 0.75 || abs(k_r[2]) > 0.75 || abs(k_r[3]) > 0.75
        println("k in FCC basis is > 0.75")
#         println(k_r)
#         println(FCC_to_cartesian_recip * k_r)
#         println(in_1BZ_FCC(FCC_to_cartesian_recip * k_r))
        return 0
    end
    
#     BCC_basis_k = cartesian_to_BCC_recip * reshape(k0, 3, 1)
    
    if abs(offset[1]) > 3 || abs(offset[2]) > 3 || abs(offset[3]) > 3
#         println("offset is too much")
#         println(offset)
        return 0
    end
    
    n1, n2, n3 = -offset .+ [4,4,4]
    g_fcc = cartesian_to_FCC_recip * gvec
#     println(g_fcc)
    Msplines = [D.Mspline[band, Integer(n1), Integer(n2), Integer(n3),i] for i in 1:3]
#     println(Msplines)
    ωspline = D.ωspline[band]

    ω0 = ωspline(k_r[1], k_r[2], k_r[3])
    M02 = reshape([Msplines[i](k_r[1], k_r[2], k_r[3])^2 for i in 1:3], 1, 3) * reshape(normalize([1,1,1]), 3, 1)
#     M02 = reshape([Msplines[i](k_r[1], k_r[2], k_r[3])^2 for i in 1:3], 1, 3) * reshape(normalize(g_fcc)*sqrt(3), 3, 1)

    M02 = M02[1]

    ωi = ω0 * 2π*c / (sqrt(ϵbg)*a)

    #return func
    
    Ep2 = ħ*np*ωp/(2*ϵ0)
    prefactor = ωs^3/(16*π^2*c^3*ϵbg)*Ep2 * L^3 * χ^2
    
    
    return prefactor * ωi * M02 * lor(ωp - ωs - ωi, lz)
    
        
end


function info_stitch_FCC(ωs, θs, ϕs, gvec, P::PumpBeam, D::PhC_Data, χ, np, L, band, ϵbg=1, lz=0.2e14)
    ωp = P.ωp
    θp = P.θp 
    ϕp = P.ϕp
    
    kp = ωp/c
    ks = ωs/c
    
    a = D.a
#     g = gvec[1] #assuming in the x direction
    
    kBx = -ks*sin(θs) - kp*sin(θp) + gvec[1]
    kBy = -ks*cos(θs)*sin(ϕs) + kp*cos(θp)*sin(ϕp) + gvec[2]
    kBz = -ks*cos(θs)*cos(ϕs) + kp*cos(θp)*cos(ϕp) + gvec[3]
    
    kB0x = kBx * a / (2π)
    kB0y = kBy * a / (2π)
    kB0z = kBz * a / (2π)
    
    # k_r and offset both in recip lattice basis
    k_r, offset = bring_into_1BZ_FCC(kB0x, kB0y, kB0z)
#     k_r0 = cartesian_to_BCC_recip * [kB0x, kB0y, kB0z]
#     k_r = collect(ms.first_brillouin_zone(mp.Vector3(k_r0[1], k_r0[2], k_r0[3])))
#     offset = k_r .- k_r0
    kcart = FCC_to_cartesian_recip * k_r
    if abs(k_r[1]) > 0.75 || abs(k_r[2]) > 0.75 || abs(k_r[3]) > 0.75
#         println("k in FCC basis is > 0.75")
        println(k_r)
#         println(FCC_to_cartesian_recip * k_r)
        println("is it in the 1BZ: ", in_1BZ_FCC(FCC_to_cartesian_recip * k_r))
        return [0,0,0],[0,0,0],[0,0,0],0,0,0,0
    end
    
#     BCC_basis_k = cartesian_to_BCC_recip * reshape(k0, 3, 1)
    
    if abs(offset[1]) > 3 || abs(offset[2]) > 3 || abs(offset[3]) > 3
        println("offset is too much")
        println(offset)
        println("k_r: ", k_r)
        println(kB0x)
        println(kB0y)
        println(kB0z)
        return [0,0,0],[kB0x, kB0y, kB0z],[0,0,0],0,0,0,0
    end
    
    n1, n2, n3 = -offset .+ [4,4,4]
    g_fcc = cartesian_to_FCC_recip * gvec
#     println(g_fcc)
#     Msplines = [D.Mspline[band, Integer(n1), Integer(n2), Integer(n3),i] for i in 1:3]
#     println(Msplines)
    ωspline = D.ωspline[band]

    ω0 = ωspline(k_r[1], k_r[2], k_r[3])
#     M0 = reshape([Msplines[i](k_r[1], k_r[2], k_r[3]) for i in 1:3], 1, 3) * reshape(g_fcc, 3, 1)
#     M0 = M0[1]
    
    M0 = D.Mspline[band, Integer(n1), Integer(n2), Integer(n3), 1](k_r[1], k_r[2], k_r[3])
    M02 = sum([D.Mspline[band, Integer(n1), Integer(n2), Integer(n3), i](k_r[1], k_r[2], k_r[3]) for i in 1:3].*normalize(g_fcc))^2
    M022 = sum([D.Mspline[band, Integer(n1), Integer(n2), Integer(n3), i](k_r[1], k_r[2], k_r[3]) for i in 1:3].^2)
    ωi = ω0 * 2π*c / (sqrt(ϵbg)*a)

    #return func
    
    Ep2 = ħ*np*ωp/(2*ϵ0)
    prefactor = ωs^3/(16*π^2*c^3*ϵbg) * Ep2 * L^3 * χ^2
    
    return offset, [kB0x, kB0y, kB0z], k_r, M0, ωi, lor(ωp - ωs - ωi, lz), M02, M022
end



###################################
###################################

function dηdω(Nω::Integer, Emax::Float64, P::PumpBeam, D::PhC_Data, lattice::String, gvec, χ, np, L; Nangles=100, Nb=12, Emin=0.01, θdev=0.00225, ϕdev=0.0025, lz=0.2e14, pol="te")
    if lattice == "sc"
        dΓdωdΩ = dΓdωdΩ_SC
        j = 1
    elseif lattice == "sc_fs"
        dΓdωdΩ = dΓdωdΩ_SC_fs
        j = 1
    elseif lattice == "bcc"
        dΓdωdΩ = dΓdωdΩ_stitch_BCC
        j = 1/2
    elseif lattice == "fcc"
        dΓdωdΩ = dΓdωdΩ_stitch_FCC
        j = 1/4
    elseif lattice == "1D"
        dΓdωdΩ = dΓdωdΩ_1D
        j = 1
    elseif lattice == "1D_a"
        dΓdωdΩ = dΓdωdΩ_1D_analytical_1
        j = 1
    else
        println("Invalid lattice, choose from sc, bcc, and fcc")
        return
    end
    θp = P.θp
    ϕp = P.ϕp
    θB = asin(norm(gvec)*c/(2*ωp))
    θrange = range(2*θB - θp - θdev, 2*θB - θp + θdev, length=Nangles)
    ϕdev = 0.00175/cos(2*θB - θp)
    ϕrange = range(ϕp -ϕdev, ϕp + ϕdev, length=Nangles)
    println("phidev = $(ϕdev)")
    println("θp = $(θp)")
    Δθ = θrange[2]-θrange[1]
    Δϕ = ϕrange[2]-ϕrange[1]

    ωirange = range(Emin/ħeV, Emax/ħeV, length=Nω)
    dηdω = zeros(Nω)
    
    
    
    for (i, ωi) in enumerate(ωirange)
        println("omega i = ", string(i))
        global dat = zeros(Nangles, Nangles)
        Threads.@threads for b in 1:Nb
            if lattice == "1D" || lattice == "1D_a"
                newdat = [dΓdωdΩ(ωp - ωi, θs, ϕs, gvec, P, D, χ, np, L, b, lz=lz, pol=pol)*cos(θs)
                                    for θs in θrange, ϕs in ϕrange]
            else
                newdat = [dΓdωdΩ(ωp - ωi, θs, ϕs, gvec, P, D, χ, np, L, b, lz=lz)*cos(θs)
                                    for θs in θrange, ϕs in ϕrange]
            end
            global dat = dat .+ newdat
        end
        dηdω[i] = j*sum(dat)*Δθ*Δϕ/(np*c*L^2)
    end
    return dηdω
end

function dηdω_free(Nω::Integer, Emax::Float64, ϵ, P::PumpBeam, gvec, χ, np, L; Nangles=400, θdev=0.00125, ϕdev=0.0025, lz=0.2e14, pol="te")
    θp = P.θp
    ϕp = P.ϕp
    θB = asin(norm(gvec)*c/(2*ωp))
    θrange = range(2*θB - θp - θdev, 2*θB - θp + θdev, length=Nangles)
    ϕdev = 0.00175/cos(2*θB - θp)
    ϕrange = range(ϕp -ϕdev, ϕp + ϕdev, length=Nangles)
    Δθ = θrange[2]-θrange[1]
    Δϕ = ϕrange[2]-ϕrange[1]

    ωirange = range(0.01/ħeV, Emax/ħeV, length=Nω)
    dηdω = zeros(Nω)
    dat2 = zeros(Nangles, Nangles)
    for (i, ωi) in enumerate(ωirange)
        println("omega i = ", string(i))
        dat = [dΓdωdΩ_free(ωp - ωi, θs, ϕs, gvec, P, ϵ, χ, np, L, lz=lz, pol=pol)*cos(θs)
                                for θs in θrange, ϕs in ϕrange]
        dηdω[i] = sum(dat)*Δθ*Δϕ/(np*c*L^2)
        dat2 = dat
    end
    return dηdω, dat2, Δθ, Δϕ, θrange
end


###########
########### Plotting #############
                     #############
function ang_spec_free(Ei, P, ϵbg, gvec, χ, np, L; Nangles=100, ωp=1e19, θdev=0.0015, ϕdev=-1, lz=0.2e14, pol="both")
    ωi = Ei/ħeV
    θB = asin(norm(gvec)*c/(2*ωp))

    θrange = range(2*θB - P.θp - θdev, 2*θB - P.θp + θdev, length=Nangles)
    if ϕdev< 0
        ϕdev = 0.00175/cos(2*θB - P.θp)
    end
    ϕrange = range(P.ϕp -ϕdev, P.ϕp + ϕdev, length=Nangles)

    Δθ = θrange[2]-θrange[1]
    Δϕ = ϕrange[2]-ϕrange[1]
    if pol == "both"
        dat = [(dΓdωdΩ_free(ωp - ωi, θs, ϕs, gvec, P, ϵbg, χ, np, L, lz=lz, pol="te")*cos(θs)
            + dΓdωdΩ_free(ωp - ωi, θs, ϕs, gvec, P, ϵbg, χ, np, L, lz=lz, pol="tm")*cos(θs))
                            for θs in θrange, ϕs in ϕrange]
    elseif pol == "te"
        dat = [dΓdωdΩ_free(ωp - ωi, θs, ϕs, gvec, P, ϵbg, χ, np, L, lz=lz, pol="te")*cos(θs) for θs in θrange, ϕs in ϕrange]
    elseif pol == "tm"
        dat = [dΓdωdΩ_free(ωp - ωi, θs, ϕs, gvec, P, ϵbg, χ, np, L, lz=lz, pol="tm")*cos(θs) for θs in θrange, ϕs in ϕrange]
    else
        dat = zeros(Nangles, Nangles)
    end
    return dat, θrange, ϕrange
end
    
function ang_spec(Ei, lattice::String, D::PhC_Data, P::PumpBeam, gvec, χ, np, L; Nangles=100, Nb=12, ωp=1e19, θdev=0.00225, lz=0.2e14, ϕdev=-1, g=3.1e10)        
    if lattice == "sc"
        dΓdωdΩ = dΓdωdΩ_SC
        j = 1
    elseif lattice == "bcc"
        dΓdωdΩ = dΓdωdΩ_stitch_BCC
        j = 1/2
    elseif lattice == "fcc"
        dΓdωdΩ = dΓdωdΩ_stitch_FCC
        j = 1/4
    elseif lattice == "1D"
        dΓdωdΩ = dΓdωdΩ_1D
        j = 1
    else
        println("Invalid lattice, choose from sc, bcc, and fcc")
        return
    end
        
    ωi = Ei/ħeV
    θB = asin(g*c/(2*ωp))

    global dat = zeros(Nangles,Nangles)
    
    θrange = range(2*θB - P.θp - θdev, 2*θB - P.θp + θdev, length=Nangles)
    if ϕdev < 0
        ϕdev = 0.00175/cos(2*θB - P.θp)
    end
    ϕrange = range(P.ϕp -ϕdev, P.ϕp + ϕdev, length=Nangles)

    Δθ = θrange[2]-θrange[1]
    Δϕ = ϕrange[2]-ϕrange[1]

    # noffset, k0_vec_no_unit, k_r, M0, ωi, lor
    Threads.@threads for b in 1:Nb
        newdat = [dΓdωdΩ(ωp - ωi, θs, ϕs, gvec, P, D, χ, np, L, b, lz=lz)*cos(θs)
                            for θs in θrange, ϕs in ϕrange]
        global dat .+= newdat
    end
    return j * dat, θrange, ϕrange
end

function BZ_boundaries(Ei, lattice::String; Nangles::Integer=300)
    ωs = ωp - Ei/ħeV
    colors1 = zeros(Nangles, Nangles)
    colors2 = zeros(Nangles, Nangles)
    colors3 = zeros(Nangles, Nangles)

    for (i, θs) in enumerate(θrange)
        for (j, ϕs) in enumerate(ϕrange)
            kB0x = (-ωs/c*sin(θs)*cos(ϕs) - ωp/c*sin(θB) + gvec[1])* a/(2π)
            kB0y = -ωs/c*sin(θs)*sin(ϕs)*a/(2π)
            kB0z = (-ωs/c*cos(θs) + ωp/c*cos(θB))*a/(2π)
            offset = bring_into_1BZ_FCC(kB0x, kB0y, kB0z)[2]
            colors1[i, j] = offset[1]
            colors2[i, j] = offset[2]
            colors3[i, j] = offset[3]

        end
    end
end



############### Reading MPB mode files ################

function read_phc_data(fname::String, lattice::String, a::AbstractFloat)
    data = npzread(fname)

    kBZ = data["arr_0"]
    Δk = kBZ[2] - kBZ[1]
    if lattice == "SC" || lattice == "SC_fs" || lattice == "SC_fs_nn"
        krange = -0.5:Δk:0.5
        ks = range(-0.5,0.5,length=70)
    else
        krange = -0.75:Δk:0.75
        ks = range(-0.75,0.75,length=70)
    end

    M1dat = data["arr_1"]
    M2dat = data["arr_2"]
    M3dat = data["arr_3"]
    ωdat = data["arr_4"]
    vgdat = data["arr_5"];

    Nb = size(ωdat)[1]
    NM = size(M1dat)[5]
    Nk = size(ωdat)[2]
    ωsplines = Array{Any,1}(undef, Nb)
    k0 = Integer((Nk+1)/2) # Nk must be odd

    for b in 1:Nb
        ωband = ωdat[b,:,:,:]
        ωitp = interpolate(ωband, BSpline(Quadratic(Flat(OnCell()))));
        ωsitp = scale(ωitp, krange, krange, krange);
        ωsplines[b] = ωsitp
    end

    Msplines = Array{Any,5}(undef, Nb, NM, NM, NM, 3)
    for i in 1:3
        key = string("arr_", i)
        Mdat = data[key]
        for b in 1:Nb
            for nx in 1:NM
                for ny in 1:NM
                    for nz in 1:NM
                        Mitp = interpolate(Mdat[b,:,:,:,nx,ny,nz], BSpline(Cubic(Flat(OnCell()))));
                        Msitp = scale(Mitp, krange, krange, krange);
                        Msplines[b,nx,ny,nz,i] = Msitp
                    end
                end
            end
        end
    end

    return PhC_Data(a, Msplines, ωsplines, 0), ωdat, Nk;
end