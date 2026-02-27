# Changelog

All notable changes to MineLab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-02-25

### Added

#### Utilities (`minelab.utilities`)
- Unit conversions: length, mass, volume, density, pressure, angle, energy, flow rate
- Mineral database with 50+ minerals (density, hardness, formula, color)
- Input validators: positive, non-negative, range, percentage, array
- Grade handling: ppm/pct/opt conversions, cutoff filtering, dilution, NSR
- Descriptive statistics: mean, std, CV, skew, kurtosis, quantiles, capping
- Visualization helpers: histogram, scatter, variogram plot, box plot

#### Economics (`minelab.economics`)
- Cash flow analysis: NPV, IRR, payback period, equivalent annual annuity
- Cost models: CAPEX estimation by capacity, OPEX breakdown
- Monte Carlo simulation with triangular/normal distributions
- Sensitivity analysis (spider/tornado diagrams)

#### Data Management (`minelab.data_management`)
- Drillhole loading/saving (CSV, Excel)
- Compositing: by length, by bench, by geology
- Desurvey: minimum curvature, tangential methods
- I/O format support: CSV, GeoJSON

#### Geostatistics (`minelab.geostatistics`)
- Experimental variograms: omnidirectional, directional, cross, cloud
- Variogram models: spherical, exponential, Gaussian, power, nugget, hole effect, nested
- Model fitting: WLS, manual, auto-fit (best-of-3)
- Kriging: ordinary, simple, universal, indicator, block
- Cross-validation (leave-one-out)
- Simulation: sequential Gaussian (SGS), sequential indicator (SIS)
- Back-transform and simulation statistics (E-type, P10/P50/P90)
- Transformations: normal score, log, indicator
- Declustering: cell, polygonal
- Block model: creation, grade-tonnage curves

#### Geomechanics (`minelab.geomechanics`)
- Rock mass classification: RMR (Bieniawski 1989), Q-system (Barton 1974), GSI
- Hoek-Brown criterion: intact and rock mass strength, deformation modulus
- Slope stability: Bishop simplified, planar failure, wedge (Markland test)
- Support design: bolt spacing, shotcrete thickness, Barton support pressure

#### Mineral Processing (`minelab.mineral_processing`)
- Comminution: Bond work index, mill power, crushing energy
- Flotation: rate constant, recovery, bank design
- Leaching: shrinking core model, heap leach recovery, tank residence time
- Classification: GGS, RR particle distributions, Lynch-Rao partition curve
- Thickening: Kynch settling, Talmage-Fitch sizing
- Gravity separation: Falcon concentrator, jig recovery, DMS efficiency
- Magnetic separation: Davis tube recovery, WHIMS, mineral susceptibility
- Mass balance: two-product formula, three-product formula, closure check

#### Drilling & Blasting (`minelab.drilling_blasting`)
- Blast design: burden (Langefors/Konya), spacing, stemming, subgrade, powder factor
- Fragmentation: Kuz-Ram, modified Kuz-Ram, SWEBREC distribution, uniformity index
- Vibration: PPV scaled distance, USBM method, compliance check
- Flyrock: range estimation, safety distance
- Blastability: Lilly index, rock factor

#### Ventilation (`minelab.ventilation`)
- Airway resistance: Atkinson equation, friction factor, series/parallel
- Network solving: Hardy Cross iterative method
- Fan selection: operating point, power calculation, series/parallel combinations
- Gas dilution: diesel, blasting fumes, methane, dust
- Similarity laws: fan affinity, specific speed

#### Equipment (`minelab.equipment`)
- Truck cycle: cycle time, rimpull speed, travel time
- Fleet matching: match factor, optimal fleet size
- Productivity: fleet, excavator, OEE
- Fuel consumption: rate and cost per tonne

#### Mine Planning (`minelab.mine_planning`)
- Pit optimization: Lerchs-Grossmann 2D, pseudoflow 3D, block economic value
- Cut-off grade: breakeven, Lane's method, marginal
- Scheduling: greedy period assignment, NPV schedule, precedence constraints
- Pushbacks: nested pit shells, design pushbacks
- Mine design: pit geometry, ramp design, volume-tonnage
- Reserves: resource-to-reserve conversion, dilution and ore loss

#### Production (`minelab.production`)
- Blending: LP optimization, grade calculation
- Grade control: SMU classification, information effect
- Stockpiles: FIFO and LIFO management
- Reconciliation: F-factors, variance analysis, reconciliation report

#### Resource Classification (`minelab.resource_classification`)
- JORC 2012: classification and Table 1 generation
- NI 43-101: classification with CIM standards
- Classification criteria: kriging variance, search pass, slope of regression
- Reporting: resource statement, grade-tonnage by category

#### Environmental (`minelab.environmental`)
- Acid mine drainage: MPA, ANC, NAPP, NAG classification, paste pH prediction
- Water balance: site balance, pit dewatering (Darcy), runoff (rational method)
- Tailings: storage capacity, beach angle estimation
- Dust: AP-42 emission factors, Gaussian plume dispersion

#### Examples
- 8 example scripts demonstrating complete workflows
- End-to-end mine workflow example

#### Integration Tests
- Cross-module workflow tests
- Mass balance closure verification
- Data I/O round-trip tests
- Resource estimation pipeline tests
