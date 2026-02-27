# MineLab

**Librería integral de ingeniería minera y metalúrgica para Python.**

MineLab proporciona herramientas de cálculo para todas las disciplinas de la ingeniería de minas: geoestadística, planeamiento minero, geomecánica, procesamiento de minerales, perforación y voladura, ventilación, economía minera, medio ambiente y más.

## Instalación

```bash
pip install minelab
```

Para desarrollo:

```bash
git clone https://github.com/minelab/minelab.git
cd minelab
py -m uv sync --extra dev
```

## Inicio Rápido

```python
import minelab as ml

# Calcular NPV de un proyecto minero
resultado = ml.npv(rate=0.10, cashflows=[-100_000, 30_000, 35_000, 40_000, 45_000])
print(f"NPV: ${resultado:,.0f}")

# Clasificación de macizo rocoso
rmr = ml.rmr_bieniawski(
    ucs_rating=12, rqd_rating=17, spacing_rating=10,
    condition_rating=20, groundwater_rating=10, orientation_adj=-5,
)
print(f"RMR: {rmr}")

# Ensayo de bombeo — Theis
descenso = ml.theis_drawdown(Q=500, T=150, S=0.001, r=50, t=1)
print(f"Descenso: {descenso:.2f} m")

# Diseño de caserón subterráneo
estabilidad = ml.mathews_stability(q_prime=8, a=0.8, b=0.3, c=4)
print(f"N' = {estabilidad['n_prime']:.1f} → {estabilidad['stability_zone']}")
```

También puedes importar desde submódulos específicos:

```python
from minelab.economics import npv, irr, tornado_analysis
from minelab.geostatistics import ordinary_kriging, experimental_variogram
```

## Módulos

| Módulo | Descripción |
|--------|-------------|
| `utilities` | Conversiones de unidades, base de datos de minerales, validadores, estadística |
| `geostatistics` | Variogramas, kriging, simulación, modelos de bloques |
| `mine_planning` | Optimización de pit, ley de corte, scheduling |
| `geomechanics` | Clasificación de macizo rocoso, estabilidad de taludes, soporte |
| `mineral_processing` | Conminución, flotación, lixiviación, balance metalúrgico |
| `equipment` | Ciclo de camiones, match factor, productividad de flota |
| `drilling_blasting` | Diseño de voladura, fragmentación Kuz-Ram, vibración |
| `ventilation` | Resistencia de vías, Hardy Cross, selección de ventiladores |
| `economics` | VAN, TIR, Monte Carlo, análisis de sensibilidad |
| `environmental` | Drenaje ácido, balance hídrico, relaves |
| `production` | Blending, control de leyes, stockpiles |
| `resource_classification` | JORC 2012, NI 43-101 |
| `data_management` | Sondajes, compositos, desurvey, formatos I/O |
| `underground_mining` | Caserones, convergencia-confinamiento, room & pillar, relleno |
| `hydrogeology` | Ensayos de acuífero, desaguado de rajo, química de aguas |
| `surveying` | Volumetría, coordenadas UTM, topografía de tronaduras |

## Requisitos

- Python >= 3.10
- NumPy, SciPy, Pandas, Matplotlib, Plotly

## Licencia

MIT License — ver [LICENSE](LICENSE).
