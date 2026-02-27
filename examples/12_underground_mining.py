"""Minería subterránea: diseño de caserón, convergencia-confinamiento y relleno.

Este ejemplo demuestra un flujo de trabajo completo para una operación
subterránea por subniveles con relleno cementado:

1. Evaluar estabilidad del caserón (Mathews).
2. Dimensionar el caserón según restricción de radio hidráulico.
3. Calcular curva de reacción del terreno (GRC) y presión de soporte.
4. Diseñar el relleno cementado y su resistencia requerida.
5. Estimar el valor económico del caserón.
"""

from __future__ import annotations

from minelab.geomechanics import (
    cable_bolt_capacity,
    in_situ_stress_depth,
    kirsch_elastic_stress,
)
from minelab.mine_planning import (
    crown_pillar_thickness,
    stope_economic_value,
    underground_cutoff_grade,
)
from minelab.underground_mining import (
    arching_stress,
    backfill_requirement,
    cemented_paste_strength,
    ground_reaction_curve,
    hydraulic_radius,
    mathews_stability,
    mucking_rate,
    stope_dimensions,
    sublevel_interval,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parámetros del caserón
    # ------------------------------------------------------------------
    ore_width = 15.0  # m ancho del cuerpo mineralizado
    ore_dip = 70.0  # grados
    stope_height = 30.0  # m altura de caserón
    hr_limit = 10.0  # límite de radio hidráulico
    depth = 400.0  # m profundidad

    print("=== Parámetros del Caserón ===")
    print(f"  Ancho de mineral:   {ore_width} m")
    print(f"  Manteo:             {ore_dip} grados")
    print(f"  Altura de caserón:  {stope_height} m")
    print(f"  Profundidad:        {depth} m")

    # ------------------------------------------------------------------
    # 2. Estabilidad de Mathews
    # ------------------------------------------------------------------
    stab = mathews_stability(q_prime=8.0, a=0.8, b=0.3, c=4.0)
    print("\n=== Estabilidad de Mathews ===")
    print(f"  N' (número de estabilidad): {stab['n_prime']:.2f}")
    print(f"  Zona de estabilidad:        {stab['stability_zone']}")

    # Radio hidráulico
    hr = hydraulic_radius(30.0, stope_height)
    print(f"  Radio hidráulico:           {hr:.2f} m")

    # Dimensiones del caserón
    dims = stope_dimensions(ore_width, ore_dip, stope_height, hr_limit)
    print(f"  Strike length máximo:       {dims['max_strike_length']:.1f} m")
    print(f"  HR real:                    {dims['actual_hr']:.2f} m")

    # ------------------------------------------------------------------
    # 3. Estado tensional y convergencia
    # ------------------------------------------------------------------
    stress = in_situ_stress_depth(depth, density=2700.0, k_ratio=1.5)
    print("\n=== Estado Tensional ===")
    print(f"  sigma_v:    {stress['sigma_v_mpa']:.2f} MPa")
    print(f"  sigma_h:    {stress['sigma_h_mpa']:.2f} MPa")

    # Kirsch alrededor de excavación circular equivalente
    kirsch = kirsch_elastic_stress(
        sigma_v=stress["sigma_v_mpa"],
        sigma_h=stress["sigma_h_mpa"],
        r_tunnel=ore_width / 2,
        r=ore_width / 2,
        theta_deg=90,
    )
    print(f"  sigma_tangencial (corona): {kirsch['sigma_tangential']:.2f} MPa")

    # GRC
    grc = ground_reaction_curve(
        p_i=0.0,
        sigma_0=stress["sigma_v_mpa"],
        sigma_ci=120.0,
        mi=15.0,
        gsi=65.0,
        r_tunnel=ore_width / 2,
        e_rock=30000.0,
    )
    print("\n=== Curva de Reacción del Terreno ===")
    print(f"  Presión crítica:    {grc['p_critical_mpa']:.2f} MPa")
    print(f"  Convergencia máx:   {grc['convergence_pct']:.2f} %")

    # ------------------------------------------------------------------
    # 4. Soporte con cable bolts
    # ------------------------------------------------------------------
    cable = cable_bolt_capacity(
        diameter_mm=15.2,
        ucs_grout=40.0,
        embedment_length=3.0,
    )
    print("\n=== Cable Bolts ===")
    print(f"  Capacidad de diseño: {cable['design_capacity_kn']:.1f} kN")

    # ------------------------------------------------------------------
    # 5. Relleno cementado
    # ------------------------------------------------------------------
    ore_volume = dims["max_strike_length"] * ore_width * stope_height
    bfr = backfill_requirement(ore_volume, void_filling_ratio=0.95, fill_density=2.0)
    print("\n=== Relleno ===")
    print(f"  Volumen de caserón:     {ore_volume:,.0f} m3")
    print(f"  Volumen de relleno:     {bfr['fill_volume_m3']:,.0f} m3")
    print(f"  Masa de relleno:        {bfr['fill_mass_tonnes']:,.0f} t")

    paste_ucs = cemented_paste_strength(
        cement_content=0.05,
        cure_days=28,
        water_cement_ratio=7.0,
    )
    print(f"  UCS pasta (28 días):    {paste_ucs:.0f} kPa")

    arch = arching_stress(
        fill_height=stope_height,
        fill_width=ore_width,
        cohesion=20.0,
        friction_angle=30.0,
        density=2000.0,
    )
    print(f"  Tensión vertical relleno: {arch['vertical_stress_kpa']:.1f} kPa")
    print(f"  Ratio de arqueo:          {arch['arching_ratio']:.3f}")

    # ------------------------------------------------------------------
    # 6. Sublevel interval
    # ------------------------------------------------------------------
    si = sublevel_interval(ore_dip, draw_angle=70.0, burden=3.0)
    print("\n=== Subniveles ===")
    print(f"  Intervalo de subnivel: {si:.1f} m")

    # ------------------------------------------------------------------
    # 7. Productividad LHD
    # ------------------------------------------------------------------
    lhd_rate = mucking_rate(
        bucket_capacity=6.0,
        fill_factor=0.85,
        cycle_time_min=4.0,
        density=2.7,
    )
    print(f"  Productividad LHD: {lhd_rate:.1f} t/h")

    # ------------------------------------------------------------------
    # 8. Evaluación económica del caserón
    # ------------------------------------------------------------------
    ore_tonnes = ore_volume * 2.7
    econ = stope_economic_value(
        ore_tonnes=ore_tonnes,
        grade=3.5,
        metal_price=50.0,
        recovery=0.92,
        opex_per_tonne=45.0,
        dilution=0.10,
    )
    print("\n=== Evaluación Económica ===")
    print(f"  Tonelaje mineral:     {ore_tonnes:,.0f} t")
    print(f"  Ley diluida:          {econ['diluted_grade']:.2f}")
    print(f"  Ingreso:              USD {econ['revenue']:,.0f}")
    print(f"  Costo:                USD {econ['cost']:,.0f}")
    print(f"  Utilidad:             USD {econ['profit']:,.0f}")
    print(f"  Utilidad/t:           USD {econ['profit_per_tonne']:.2f}")

    cog = underground_cutoff_grade(
        opex_per_tonne=45.0,
        price=50.0,
        recovery=0.92,
        mining_cost_ug=25.0,
    )
    print(f"  Ley de corte UG:      {cog:.2f}")

    # Corona
    crown = crown_pillar_thickness(
        span=ore_width,
        rock_density=2700.0,
        sigma_cm=25.0,
        safety_factor=2.0,
    )
    print(f"  Espesor de pilar corona: {crown:.1f} m")


if __name__ == "__main__":
    main()
