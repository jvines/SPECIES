#!/usr/bin/env julia
# Solve atmospheric parameters from a SPECIES MOOG-format line list using Korg.
#
#   julia --project=<KLOTHO> solve_ews.jl <linelist.txt> <out.json> [key=value ...]
#
# This is the Julia half of SPECIES's second engine. SPECIES measures the
# equivalent widths and writes the MOOG line list; this reads that same file and
# solves the four classical excitation/ionisation conditions with Korg + MARCS,
# where the primary engine uses MOOG + ATLAS9. Both engines therefore consume
# byte-identical equivalent widths, so any difference between them is radiative
# transfer and atmosphere grid rather than measurement.
#
# Communication is a file and a JSON document, not an in-process binding.
# juliacall would embed a Julia runtime in CPython and hold the GIL for the
# duration of every call, which is disqualifying for a pipeline whose workload
# is one independent star per core. MOOG is already driven as a subprocess, so
# this is the same shape.
#
# Options (key=value):
#   teff0, logg0, feh0, vt0    starting guess
#   hold                       comma-separated: temperature,gravity,metallicity,velocity
#   sigma_clip                 true|false  (default true)
#   max_outer                  integer     (default 3)

using Pkg
using KLOTHO
# railed_parameters is public-but-unexported in KLOTHO. Julia resolves names at
# call time, so omitting it here let the whole solve run and then die while
# assembling the result -- minutes of Korg synthesis thrown away for a missing
# import. `preflight()` below touches every such symbol before any work starts.
import KLOTHO: solve_with_covariance, railed_parameters
using DelimitedFiles
using LinearAlgebra
using JSON3
using Printf

function parse_opts(args)
    o = Dict{String,String}()
    for a in args
        occursin('=', a) || continue
        k, v = split(a, '=', limit=2)
        o[String(k)] = String(v)
    end
    o
end

optf(o, k, d) = haskey(o, k) ? parse(Float64, o[k]) : d
opti(o, k, d) = haskey(o, k) ? parse(Int, o[k]) : d
optb(o, k, d) = haskey(o, k) ? lowercase(o[k]) in ("1", "true", "yes") : d

"""
    read_moog_ew_linelist(path) -> EWTable

Read the line list `species.ew.create_moog_linelist` writes:

    <1 header line>
      wavelength   species   excitation   log gf   EW(mA)

Column order differs from the SPECIES *atomic* line list that
`KLOTHO.read_species_linelist` reads, which is why this is here rather than
there.

`ew_err` is filled with a 10% placeholder. The excitation/ionisation solve does
not consume it -- the parameter covariance is built from the line-to-line
abundance scatter, not from the input EW errors -- so it affects nothing the
caller reads. It matters only to `solve_monte_carlo`, which this does not run.
"""
function read_moog_ew_linelist(path::AbstractString)
    raw = readdlm(path; skipstart=1)
    wl = Float64.(raw[:, 1])
    elm = Float64.(raw[:, 2])
    exc = Float64.(raw[:, 3])
    gf = Float64.(raw[:, 4])
    ew = Float64.(raw[:, 5])
    keep = ew .> 0
    EWTable(wavelength=wl[keep], ew=ew[keep],
            ew_err=max.(0.1 .* ew[keep], 1.0),
            element=elm[keep], excitation=exc[keep], loggf=gf[keep])
end

function main()
    # Touch the symbols the result-assembly path needs, before any solving.
    # Julia resolves names at call time, so a missing import there costs a full
    # Korg solve before it surfaces -- which is what happened with
    # `railed_parameters`. Seconds here instead of minutes.
    railed_parameters([5777.0, 4.44, 1.0, 0.0])

    length(ARGS) >= 2 || error("usage: solve_ews.jl <linelist.txt> <out.json> [key=value ...]")
    linelist_path, out_path = ARGS[1], ARGS[2]
    o = parse_opts(ARGS[3:end])

    payload = Dict{String,Any}("engine" => "korg+marcs",
                               "klotho_version" => string(pkgversion(KLOTHO)))
    try
        ews = read_moog_ew_linelist(linelist_path)
        payload["n_lines"] = length(ews)
        payload["n_fe_i"] = count(==(26.0), ews.element)
        payload["n_fe_ii"] = count(==(26.1), ews.element)

        initial = StellarParams(teff=optf(o, "teff0", 5500.0),
                                logg=optf(o, "logg0", 4.36),
                                feh=optf(o, "feh0", 0.0),
                                vt=optf(o, "vt0", 1.23))

        # SPECIES names held parameters; Korg wants a positional mask over
        # [Teff, logg, vmic, [m/H]].
        hold = haskey(o, "hold") ? split(o["hold"], ',') : String[]
        fix = [("temperature" in hold), ("gravity" in hold),
               ("velocity" in hold), ("metallicity" in hold)]
        payload["hold"] = collect(hold)

        kw = any(fix) ? (; fix_params=fix) : (;)
        fitted, pc, converged = solve_with_covariance(
            EWBalance(), ews;
            initial=initial,
            sigma_clip=optb(o, "sigma_clip", true),
            max_outer=opti(o, "max_outer", 3),
            kw...)

        railed = railed_parameters(pc.params)
        payload["teff"] = fitted.teff
        payload["logg"] = fitted.logg
        payload["feh"] = fitted.feh
        payload["vt"] = fitted.vt
        payload["converged"] = converged
        payload["railed"] = String.(railed)
        payload["retcode"] = !isempty(railed) ? "railed" :
                             (converged ? "ok" : "not_converged")
        # The reason this engine exists: the joint, not four marginals.
        C = correlation(pc)
        payload["covariance"] = Dict(
            "order" => ["teff", "logg", "vt", "feh"],
            "sigma" => sqrt.(diag(pc.Σ)),
            "matrix" => [pc.Σ[i, :] for i in 1:4],
            "correlation" => [C[i, :] for i in 1:4],
            "fixed" => String.(fixed_parameters(pc)),
            "sigma_A" => pc.σ_A,
            "n_neutral" => pc.n_lines[1],
            "n_ionised" => pc.n_lines[2],
            "korg_sigma" => pc.korg_σ,
        )
        payload["status"] = "ok"
    catch e
        payload["status"] = "error"
        payload["error"] = first(sprint(showerror, e), 400)
    end

    open(out_path, "w") do io
        JSON3.pretty(io, payload)
    end
    println("wrote ", out_path)
end

main()
