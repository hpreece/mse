# Multiple Stellar Evolution (MSE) -- A Population Synthesis Code for Multiple-Star Systems #

MSE is code that models the long-term evolution of hierarchical multiple-star systems (binaries, triples, quadruples, and higher-order systems) from the main sequence until remnant phase. It takes into account gravitational dynamical evolution, stellar evolution (using the `sse` tracks), and binary interactions (such as mass transfer and common-envelope evolution).  It includes routines for external perturbations from flybys in the field, or (to limited extent) encounters in dense stellar systems such as galactic nuclei.

C++ and Fortran compilers are required, as well as Python (2/3) for the Python interface. Make sure to first compile the code using `make`. Please modify the Makefile according to your installation (`CC`, `CXX`, and `FC` should be correctly assigned).

The script `test_mse.py` can be used to test the installation. The script `run_system.py` is useful for quickly running a system.

**See the user guide (doc/doc.pdf) for more detailed information.**

---

## Feature Reference

### Vector Resonant Relaxation (VRR)

VRR models stochastic orbit-plane precession driven by the fluctuating torques of a stellar background (e.g., near a galactic nucleus). It adds secular perturbations directly to the angular momentum vector of the chosen orbit.

**Enabling VRR:**

```python
code = MSE()
code.enable_VRR = True   # must be True to push VRR parameters to C++
```

**Particle-level parameters** (set on binary/orbital nodes, not body particles):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `VRR_model` | int | 0 | VRR model selector (0=off, 1=torque, 3=Bar-Or) |
| `VRR_include_mass_precession` | int | 0 | Add mass precession (1=yes, 0=no) |
| `VRR_mass_precession_rate` | float | 0.0 | Mass precession rate (rad/yr) |
| `VRR_initial_time` | float | 0.0 | Start of VRR interpolation window (yr) |
| `VRR_final_time` | float | 1.0 | End of VRR interpolation window (yr) |

**Model 1 — Torque (uniform precession):**

Drives `dh/dt = Ω × h`. The precession vector `Ω` is fixed in the lab frame.

| Parameter | Description |
|-----------|-------------|
| `VRR_Omega_vec_x` | x-component of precession frequency vector (rad/yr) |
| `VRR_Omega_vec_y` | y-component of precession frequency vector (rad/yr) |
| `VRR_Omega_vec_z` | z-component of precession frequency vector (rad/yr) |

Example: precess the outer orbit of a triple at 0.02 rad/yr about the x-axis:

```python
outer_orbit.VRR_model = 1
outer_orbit.VRR_include_mass_precession = 0
outer_orbit.VRR_Omega_vec_x = 0.02   # rad/yr  → T_prec ≈ 314 yr
outer_orbit.VRR_Omega_vec_y = 0.0
outer_orbit.VRR_Omega_vec_z = 0.0
```

**Model 3 — Bar-Or stochastic matrix:**

Uses a quadrupole-order stochastic torque described by five spherical-harmonic amplitudes. The amplitudes linearly interpolate between initial and final values over `[VRR_initial_time, VRR_final_time]`.

| Parameter | Description |
|-----------|-------------|
| `VRR_eta_20_init/final` | m=0, l=2 amplitude at start / end |
| `VRR_eta_a_22_init/final` | real part of m=2, l=2 amplitude |
| `VRR_eta_b_22_init/final` | imaginary part of m=2, l=2 amplitude |
| `VRR_eta_a_21_init/final` | real part of m=1, l=2 amplitude |
| `VRR_eta_b_21_init/final` | imaginary part of m=1, l=2 amplitude |

> **Note:** VRR model 2 (distortion model) is compiled out and will return error code 33 if selected. Only models 1 and 3 are functional.

---

### eCAML Mass-Transfer Stability Model

The **eCAML** (Empirical Consequential Angular Momentum Loss) model replaces the default fixed-q criterion for deciding whether a low-mass-donor mass transfer episode is stable or dynamically unstable (leading to CE).

**Reference:** Schreiber et al. (2016, MNRAS, 455, L16); stability criteria from Belloni et al. (2018, MNRAS, 478, 5639).

**Enabling eCAML:**

```python
code.binary_evolution_use_eCAML_model = True   # default: False
```

**Behaviour difference:**

| Mode | Low-mass-donor (kw ≤ 1) instability criterion |
|------|----------------------------------------------|
| Default (`False`) | `q > 0.695` (Hjellming & Webbink) |
| eCAML (`True`) | For M_donor ≤ 0.8 M☉: `ζ_RL > ζ_AD`; for M_donor > 0.8 M☉: `q > 3.0` |

where ζ_RL is the Roche-lobe mass-radius exponent and ζ_AD is the adiabatic donor mass-radius exponent (Eq. A11/A13 in Belloni+2018). The adiabatic exponent is piecewise:
- M_donor ≤ 0.38412 M☉: ζ_AD = −1/3
- M_donor > 0.38412 M☉: ζ_AD = 0.782491 − 7.46384 M_donor + 13.9255 M_donor² − 5.3892 M_donor³

eCAML is most relevant for CV-like systems (M_donor ≲ 1 M☉, compact accretor). For high-mass donors and giant donors the model falls back to standard CE criteria unchanged.

---

### LISA Band Detection

MSE can automatically halt evolution when a binary's peak gravitational-wave emission frequency crosses into the LISA band, enabling post-processing of merger candidates.

**How it works:**

The peak GW frequency is estimated each ODE step from the orbit's semi-major axis and eccentricity:

```
f_peak = f_orb × n_peak
n_peak ≈ 2 (1 - 1.01678e + 5.57372e² - 4.9271e³ + 1.68506e⁴) / (1-e²)^1.5
```

A root is detected when `f_peak` first exceeds `check_for_entering_LISA_band_critical_GW_frequency`. The evolution stops and `evolve_system()` returns with `state=5`.

**Particle-level parameters** (set on each binary/orbital particle):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `check_for_entering_LISA_band` | bool | `True` | Enable the LISA root condition for this orbit |
| `check_for_entering_LISA_band_critical_GW_frequency` | float | 31557.6 | Threshold GW frequency in yr⁻¹ (≈ 1 mHz) |
| `entering_LISA_band_has_occurred` | bool | `False` | Read-back: set True when condition fires |

**Return value mapping:**

| `state` returned by `evolve_system()` | Meaning |
|--------------------------------------|---------|
| 5 | Binary entered the LISA band |

**Log events:** When LISA-band entry is logged, the event type is `LOG_ENTER_LISA_BAND` (integer code 21). Use `code.get_log_entry(index)` to retrieve it.

**Example — detect LISA entry for a compact binary:**

```python
binary.check_for_entering_LISA_band = True
binary.check_for_entering_LISA_band_critical_GW_frequency = 31557.6  # 1 mHz

while True:
    code.evolve_system(end_time=1e10)
    if code.particles[2].entering_LISA_band_has_occurred:
        print("Binary entered LISA band at t =", code.time)
        break
    if code.time >= 1e10:
        break
```

To disable LISA-band monitoring (e.g., to save root-finding evaluations):

```python
binary.check_for_entering_LISA_band = False
```
