<div align="center">

# MineLab

### La caja de herramientas que todo ingeniero de minas merece tener en Python.

[![PyPI](https://img.shields.io/pypi/v/minelab?color=blue&label=PyPI)](https://pypi.org/project/minelab/)
[![Python](https://img.shields.io/pypi/pyversions/minelab)](https://pypi.org/project/minelab/)
[![Tests](https://img.shields.io/github/actions/workflow/status/argenismarin/minelab/ci.yml?label=tests)](https://github.com/argenismarin/minelab/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## ¿Qué es MineLab?

MineLab nació de una convicción simple: **los cálculos que hacemos los ingenieros de minas merecen algo mejor que hojas de cálculo desordenadas, scripts sueltos y fórmulas copiadas a mano.**

Soy Ingeniero de Minas y Metalurgia, y también un entusiasta de la programación. MineLab es la librería que me habría gustado tener desde el primer día de carrera — un solo lugar donde encontrar las fórmulas de Bieniawski, los modelos de Kuz-Ram, el kriging ordinario, el análisis de NPV, el diseño de flotación, y cientos de herramientas más, todas validadas, documentadas y listas para usar.

**Los números que respaldan esta v0.1.0:**

- **388 funciones** cubriendo 16 módulos de ingeniería minera
- **897 tests** con **94% de cobertura** de código
- Cada función incluye **referencias bibliográficas** a las fórmulas originales
- Compatible con **Python 3.10 – 3.13**

---

## Instalación

```bash
pip install minelab
```

---

## Inicio Rápido

```python
import minelab as ml

# --- Economía minera ---
van = ml.npv(rate=0.10, cashflows=[-100_000, 30_000, 35_000, 40_000, 45_000])
print(f"VAN del proyecto: ${van:,.0f}")

# --- Geomecánica: clasificación de macizo rocoso ---
rmr = ml.rmr_bieniawski(
    ucs_rating=12, rqd_rating=17, spacing_rating=10,
    condition_rating=20, groundwater_rating=10, orientation_adj=-5,
)
print(f"RMR89: {rmr}")

# --- Geoestadística: variograma experimental ---
from minelab.geostatistics import experimental_variogram
lags, gamma = experimental_variogram(
    coordinates=coords, values=grades, n_lags=15, lag_size=50.0
)

# --- Procesamiento de minerales: recuperación por flotación ---
rec = ml.flotation_recovery(feed_grade=1.2, concentrate_grade=28.0, tail_grade=0.15)
print(f"Recuperación: {rec:.1f}%")

# --- Perforación y voladura: fragmentación Kuz-Ram ---
x50 = ml.kuz_ram_x50(powder_factor=0.5, rock_factor=8.0, charge_weight=150.0)
print(f"Fragmentación media (x50): {x50:.1f} cm")
```

También puedes importar desde submódulos específicos:

```python
from minelab.economics import npv, irr, monte_carlo_simulation
from minelab.geostatistics import ordinary_kriging, experimental_variogram
from minelab.geomechanics import hoek_brown_failure, rmr_bieniawski
```

---

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

---

## Esto es solo el comienzo

MineLab v0.1.0 es la **primera versión pública** — la primera piedra de algo mucho más grande. En el camino vienen:

- Más funciones y módulos especializados
- Documentación completa con tutoriales en español
- Ejemplos con datos reales de la industria
- Integración con flujos de trabajo geomineros

**¿Quieres contribuir?** Toda ayuda es bienvenida: reporta bugs, sugiere funciones, o envía un pull request. Este proyecto se construye mejor en comunidad.

- [Reportar un issue](https://github.com/argenismarin/minelab/issues)
- [Ver el código fuente](https://github.com/argenismarin/minelab)

---

## Autor

**Argenis Marin** — Ingeniero de Minas y Metalurgia, entusiasta de la programación y la automatización aplicada a la minería.

[![GitHub](https://img.shields.io/badge/GitHub-argenismarin-181717?logo=github)](https://github.com/argenismarin)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Argenis%20Marin-0A66C2?logo=linkedin)](https://www.linkedin.com/in/argenismarin/)

---

## Requisitos

- Python >= 3.10
- NumPy, SciPy, Pandas, Matplotlib, Plotly

## Licencia

MIT License — ver [LICENSE](LICENSE).
