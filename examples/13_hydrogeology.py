"""Hidrogeología minera: ensayo de acuífero, desaguado y calidad de aguas.

Este ejemplo demuestra un flujo de trabajo hidrogeológico para una mina
a cielo abierto:

1. Interpretar un ensayo de bombeo con Theis y Cooper-Jacob.
2. Diseñar sistema de desaguado del rajo.
3. Evaluar calidad del agua y potencial de drenaje ácido.
4. Calcular balance de masas en punto de mezcla.
"""

from __future__ import annotations

from minelab.hydrogeology import (
    aquifer_hydraulic_conductivity,
    cone_of_depression_radius,
    cooper_jacob_drawdown,
    darcy_pit_inflow,
    dewatering_power,
    dewatering_well_capacity,
    langelier_index,
    mass_balance_water_quality,
    number_of_dewatering_wells,
    seepage_velocity,
    specific_capacity,
    theis_drawdown,
    theis_recovery,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Ensayo de bombeo
    # ------------------------------------------------------------------
    Q_pump = 500.0  # m3/dia caudal de bombeo  # noqa: N806
    T = 150.0  # m2/dia transmisividad  # noqa: N806
    S = 0.001  # coeficiente de almacenamiento  # noqa: N806
    r_obs = 50.0  # m distancia al pozo de observación
    t_obs = 1.0  # dia desde inicio del bombeo

    print("=== Ensayo de Bombeo ===")
    print(f"  Caudal de bombeo:     {Q_pump} m3/dia")
    print(f"  Transmisividad:       {T} m2/dia")
    print(f"  Coef. almacenamiento: {S}")

    s_theis = theis_drawdown(Q_pump, T, S, r_obs, t_obs)
    s_cj = cooper_jacob_drawdown(Q_pump, T, S, r_obs, t_obs)

    print(f"\n  Descenso Theis a {r_obs}m, {t_obs}d:       {s_theis:.3f} m")
    print(f"  Descenso Cooper-Jacob a {r_obs}m, {t_obs}d:  {s_cj:.3f} m")

    # Recuperación después de parar bomba
    s_rec = theis_recovery(Q_pump, T, t_pump=2.0, t_since_stop=1.0)
    print(f"  Descenso residual (t_pump=2d, t_rec=1d):  {s_rec:.3f} m")

    # Capacidad específica
    sc = specific_capacity(Q_pump, s_theis)
    print(f"  Capacidad específica: {sc:.1f} m2/dia")

    # Conductividad hidráulica
    b_acuífero = 30.0  # m espesor del acuífero
    K = aquifer_hydraulic_conductivity(T, b_acuífero)  # noqa: N806
    print(f"  K hidráulica:         {K:.2f} m/dia")

    # ------------------------------------------------------------------
    # 2. Diseño de desaguado
    # ------------------------------------------------------------------
    pit_area = 500_000.0  # m2 area del rajo
    gradient = 0.05  # gradiente hidráulico hacia el rajo

    Q_inflow = darcy_pit_inflow(K, gradient, pit_area)  # noqa: N806
    print("\n=== Desaguado del Rajo ===")
    print(f"  Influjo Darcy:        {Q_inflow:.0f} m3/dia")

    # Capacidad de cada pozo
    well_Q = dewatering_well_capacity(  # noqa: N806
        K=K,
        screen_length=20.0,
        head_reduction=15.0,
        r_well=0.15,
        r_influence=300.0,
    )
    print(f"  Capacidad por pozo:   {well_Q:.0f} m3/dia")

    # Número de pozos necesarios
    n_wells = number_of_dewatering_wells(Q_inflow, well_Q, 0.15)
    print(f"  Pozos necesarios:     {n_wells}")

    # Potencia de bombeo
    power = dewatering_power(Q_inflow, total_dynamic_head=50.0, pump_efficiency=0.7)
    print(f"  Potencia requerida:   {power:.1f} kW")

    # Cono de depresión
    R_cone = cone_of_depression_radius(K, b_acuífero, Q_inflow, t=365.0, S=S)  # noqa: N806
    print(f"  Radio cono (1 año):  {R_cone:.0f} m")

    # ------------------------------------------------------------------
    # 3. Calidad de agua
    # ------------------------------------------------------------------
    lsi = langelier_index(
        pH=7.2,
        temp_c=20.0,
        ca_ppm=150.0,
        total_alk_ppm=200.0,
        tds_ppm=500.0,
    )
    print("\n=== Calidad de Agua ===")
    print(f"  Índice de Langelier:  {lsi:.2f}")
    if lsi > 0:
        print("  -> Agua tendencia incrustante")
    else:
        print("  -> Agua tendencia corrosiva")

    # Velocidad de infiltración
    v_seep = seepage_velocity(K, gradient=0.03, porosity=0.25)
    print(f"  Velocidad infiltración: {v_seep:.3f} m/dia")

    # ------------------------------------------------------------------
    # 4. Balance de masas en punto de mezcla
    # ------------------------------------------------------------------
    flows = [Q_inflow, 200.0, 100.0]  # 3 afluentes (m3/dia)
    concentrations = [50.0, 200.0, 10.0]  # mg/L sulfatos
    mix = mass_balance_water_quality(flows, concentrations)

    print("\n=== Balance de Masas ===")
    print(f"  Caudal total:         {mix['total_flow']:.0f} m3/dia")
    print(f"  Concentración mezcla: {mix['mixed_concentration']:.1f} mg/L SO4")


if __name__ == "__main__":
    main()
