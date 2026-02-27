"""Equipment and fleet management: truck cycles, matching, and productivity.

This example demonstrates a fleet-management workflow for an open-pit mine
using CAT 793F haul trucks and a CAT 6060 hydraulic shovel:

1. **Truck cycle time** — compute haul and return segments.
2. **Rimpull & travel time** — check speed on grade.
3. **Match factor** — evaluate excavator–truck balance.
4. **Optimal fleet** — calculate required trucks for target production.
5. **Productivity** — excavator and fleet tonnage per hour, OEE.
6. **Fuel consumption** — litres per hour and cost per tonne.
"""

from __future__ import annotations

from minelab.equipment import (
    excavator_productivity,
    fleet_productivity,
    fuel_consumption_rate,
    fuel_cost_per_tonne,
    match_factor,
    oee,
    optimal_fleet,
    rimpull_speed,
    travel_time,
    truck_cycle_time,
)


def main() -> None:
    # ------------------------------------------------------------------
    # Equipment parameters (CAT 793F + CAT 6060 scenario)
    # ------------------------------------------------------------------
    truck_capacity = 227.0  # tonnes payload
    engine_kw = 1976.0  # CAT 793F engine power (kW)
    gross_weight = 384.0  # fully loaded (tonnes)
    rimpull_kn = 650.0  # rimpull at operating gear (kN)

    shovel_bucket = 34.0  # m3 bucket
    shovel_fill = 0.85  # fill factor
    shovel_cycle_sec = 28.0  # seconds per swing cycle
    material_sg = 2.6  # t/m3 bank density

    target_prod = 5000.0  # target t/h total

    print("=== Equipment Parameters ===")
    print(f"  Truck:  CAT 793F  — {truck_capacity:.0f} t payload, {engine_kw:.0f} kW")
    print(f"  Shovel: CAT 6060  — {shovel_bucket:.0f} m3 bucket")
    print(f"  Target production: {target_prod:,.0f} t/h")

    # ------------------------------------------------------------------
    # 1. Truck cycle time
    # ------------------------------------------------------------------
    print("\n=== Truck Cycle Time ===")

    haul_segments = [
        {"distance": 500, "speed": 35},  # pit ramp, loaded uphill
        {"distance": 1200, "speed": 45},  # haul road, moderate grade
        {"distance": 300, "speed": 20},  # dump approach
    ]
    return_segments = [
        {"distance": 300, "speed": 25},  # dump exit
        {"distance": 1200, "speed": 55},  # haul road, empty downhill
        {"distance": 500, "speed": 40},  # pit ramp, return
    ]

    # Loading time: bucket loads to fill truck
    loads_needed = truck_capacity / (shovel_bucket * shovel_fill * material_sg)
    load_time_min = loads_needed * shovel_cycle_sec / 60

    cycle = truck_cycle_time(
        load_time=load_time_min,
        haul_segments=haul_segments,
        dump_time=1.5,
        return_segments=return_segments,
        spot_time=0.5,
        queue_time=1.0,
    )

    print(f"  Load time:   {load_time_min:.2f} min ({loads_needed:.1f} passes)")
    print(f"  Haul time:   {cycle['haul_time']:.2f} min")
    print(f"  Return time: {cycle['return_time']:.2f} min")
    print(f"  Fixed time:  {cycle['fixed_time']:.2f} min (dump + spot + queue)")
    print(f"  TOTAL CYCLE: {cycle['total_time']:.2f} min")

    # ------------------------------------------------------------------
    # 2. Rimpull speed and travel time on grade
    # ------------------------------------------------------------------
    print("\n=== Rimpull Speed Check ===")

    grade_pct = 10.0  # 10% grade (ramp)
    rolling_r = 2.0  # 2% rolling resistance

    max_speed = rimpull_speed(rimpull_kn, grade_pct, rolling_r, gross_weight)
    print(f"  Loaded on {grade_pct}% grade: max {max_speed:.1f} km/h")

    t_ramp = travel_time(500, max_speed, grade_pct, rolling_r)
    print(f"  500 m ramp travel time: {t_ramp:.2f} min")

    # ------------------------------------------------------------------
    # 3. Match factor
    # ------------------------------------------------------------------
    print("\n=== Match Factor ===")

    loader_cycle_min = load_time_min  # time to load one truck

    for n_trucks in [4, 5, 6, 7]:
        mf = match_factor(
            n_trucks=n_trucks,
            truck_cycle=cycle["total_time"],
            n_loaders=1,
            loader_cycle=loader_cycle_min,
        )
        print(
            f"  {n_trucks} trucks: MF = {mf['mf']:.2f} "
            f"({mf['status']}, bottleneck: {mf['bottleneck']})"
        )

    # ------------------------------------------------------------------
    # 4. Optimal fleet size
    # ------------------------------------------------------------------
    print("\n=== Optimal Fleet ===")

    opt = optimal_fleet(
        truck_cycle=cycle["total_time"],
        loader_cycle=loader_cycle_min,
        target_production=target_prod,
        truck_capacity=truck_capacity,
        availability=0.85,
        utilization=0.90,
    )
    print(f"  Required trucks: {opt['n_trucks']}")
    print(f"  Expected prod:   {opt['production']:,.0f} t/h")
    print(f"  Match factor:    {opt['match_factor']:.2f}")

    # ------------------------------------------------------------------
    # 5. Productivity and OEE
    # ------------------------------------------------------------------
    print("\n=== Productivity ===")

    exc_prod = excavator_productivity(
        bucket_size=shovel_bucket,
        fill_factor=shovel_fill,
        cycle_time=shovel_cycle_sec,
        material_density=material_sg,
        availability=0.85,
    )
    print(f"  Excavator productivity: {exc_prod:,.0f} t/h")

    fleet_prod = fleet_productivity(
        n_trucks=opt["n_trucks"],
        truck_capacity=truck_capacity,
        cycle_time=cycle["total_time"],
        availability=0.85,
        utilization=0.90,
    )
    print(f"  Fleet productivity ({opt['n_trucks']} trucks): {fleet_prod:,.0f} t/h")

    equipment_oee = oee(availability=0.85, utilization=0.90, efficiency=0.95)
    print(f"  OEE: {equipment_oee:.1%}")

    # ------------------------------------------------------------------
    # 6. Fuel consumption and cost
    # ------------------------------------------------------------------
    print("\n=== Fuel Consumption ===")

    load_factor = 0.65  # average load factor for haul trucks
    fuel_rate = fuel_consumption_rate(engine_kw, load_factor)
    print(f"  Fuel rate: {fuel_rate:.1f} L/h (load factor {load_factor:.0%})")

    fuel_price = 1.10  # USD/L
    cost_per_t = fuel_cost_per_tonne(fuel_rate, fuel_price, fleet_prod)
    print(f"  Fuel cost: USD {cost_per_t:.3f} /t (@ USD {fuel_price}/L)")

    total_fleet_fuel = fuel_rate * opt["n_trucks"]
    print(f"  Total fleet fuel: {total_fleet_fuel:,.0f} L/h ({opt['n_trucks']} trucks)")


if __name__ == "__main__":
    main()
