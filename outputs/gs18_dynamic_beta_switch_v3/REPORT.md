# Dynamic β phase-switch screen

Exploratory only; the schedule was chosen from the preceding frequency screen.

Best single fixed value over the three switch paths: `beta_0.6`.

Scheduled β reduction vs best fixed: 0.742%.
Existing RL reduction vs best fixed: -0.028%.

| Method | Complete | Eligible | QP failures | Saturations (command/actual) | Mean J10 | Low-stage J10 | High-stage J10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| beta_0 | 3/3 | False | 0 | 16/0 | 0.092146737 | 0.10399703 | 0.073204513 |
| beta_0.25 | 3/3 | False | 0 | 8/0 | 0.091308338 | 0.10319953 | 0.072509641 |
| beta_0.5 | 3/3 | False | 0 | 6/0 | 0.091002724 | 0.10234772 | 0.073110032 |
| beta_0.525 | 3/3 | False | 0 | 6/0 | 0.090993296 | 0.10225951 | 0.073222607 |
| beta_0.55 | 3/3 | False | 0 | 6/0 | 0.090987089 | 0.10217067 | 0.073343294 |
| beta_0.575 | 3/3 | False | 0 | 6/0 | 0.090983986 | 0.10208119 | 0.073471801 |
| beta_0.6 | 3/3 | False | 0 | 6/0 | 0.090983868 | 0.10199109 | 0.073607836 |
| beta_0.65 | 3/3 | False | 0 | 4/0 | 0.090992241 | 0.10180957 | 0.073901441 |
| beta_0.75 | 3/3 | False | 0 | 2/0 | 0.091040526 | 0.10143985 | 0.074567994 |
| RL_beta | 3/3 | False | 0 | 5/0 | 0.091009763 | 0.10191652 | 0.07378647 |
| schedule_beta | 3/3 | False | 0 | 2/0 | 0.090309026 | 0.10142348 | 0.0725541 |
