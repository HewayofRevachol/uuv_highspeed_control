# Dynamic β phase-switch screen

Exploratory only; the schedule was chosen from the preceding frequency screen.

Best single fixed value over the three switch paths: `beta_0.25`.

Scheduled β reduction vs best fixed: -0.243%.
Existing RL reduction vs best fixed: -2.151%.

| Method | Complete | QP failures | Saturations (command/actual) | Mean J10 | Low-stage J10 | High-stage J10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beta_0 | 0/3 | 0 | 3/0 | 0.031752071 | 0.00041022741 | 0.072902513 |
| beta_0.25 | 3/3 | 0 | 0/0 | 0.031538069 | 0.00040686395 | 0.072247047 |
| beta_0.5 | 3/3 | 0 | 0/0 | 0.031875773 | 0.00040338724 | 0.072899357 |
| beta_0.525 | 3/3 | 0 | 0/0 | 0.031931755 | 0.00040302814 | 0.073017349 |
| beta_0.55 | 3/3 | 0 | 0/0 | 0.031991153 | 0.00040266685 | 0.073143472 |
| beta_0.575 | 3/3 | 0 | 0/0 | 0.032053839 | 0.00040230327 | 0.073277425 |
| beta_0.6 | 3/3 | 0 | 0/0 | 0.032119688 | 0.00040193745 | 0.073418915 |
| beta_0.65 | 3/3 | 0 | 0/0 | 0.032260425 | 0.00040119887 | 0.073723445 |
| beta_0.75 | 3/3 | 0 | 0/0 | 0.032575124 | 0.00039969373 | 0.07441181 |
| RL_beta | 3/3 | 0 | 0/0 | 0.032216307 | 0.00040200958 | 0.073646565 |
| schedule_beta | 3/3 | 0 | 0/0 | 0.031614738 | 0.00039967856 | 0.072327936 |
