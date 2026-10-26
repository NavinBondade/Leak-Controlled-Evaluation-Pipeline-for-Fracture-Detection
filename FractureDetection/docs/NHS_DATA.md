# NHS public data pulled for the hospital simulation (2026-10-25)

Files are in `data/nhs/` (downloaded, not edited; not part of the model pipeline). All figures are England-wide.

| Item | Value | Source file |
|---|---|---|
| Total A&E attendances, Sep 2026 | 2,444,743 (Type 1: 1,503,149; Type 3 / minor injury units: 886,081) | `data/nhs/ae/ae_timeseries.xls`, sheet Activity (NHS England Monthly A&E Time Series, published 8 Oct 2026) |
| Monthly attendances, Oct 2025 - Sep 2026 | 2.12 - 2.49 million | same |
| 4-hour performance, Sep 2026 | 74.4% (all types); the next column, 60.6%, is probably Type 1 only. The column header was not read, so check before use | same, sheet Performance |
| Emergency Medicine unit costs, 2025/26 | 169 (no investigation) to 1,471 per attendance depending on investigation/treatment category; 4.4 million attendances at 300 and 3.7 million at 169 | `data/nhs/ncc/national_schedule.zip` -> Emergency Care.xlsx (National Cost Collection) |
| Trauma and orthopaedic outpatient attendance, first / follow-up (face to face) | 208 / 178 per attendance | Outpatient Care.xlsx |
| Plain film in community diagnostic centres | 60.65 per examination (349,232 examinations, 39 submissions) | Diagnostic Imaging.xlsx |

## Caveats
- The plain-film unit cost above is from community diagnostic centres only. An A&E X-ray is not costed as a separate line: it is inside the Emergency Medicine investigation categories.
- No plain-radiography waiting times or reporting-delay figures were found in the files pulled. The Diagnostic Imaging Dataset and the monthly diagnostics collection (15 key tests) were not downloaded.
- Not found: a UK national missed-fracture rate. Published studies give about 1% to 4% with different definitions (see the report); use as a range, not a point.
- Costs are national averages, not per-fracture costs. Cost of a missed fracture (delayed diagnosis, recall, litigation) is not in these files and must stay an assumption.
- Licence: NHS England statistics are published under the Open Government Licence; confirm on the source page before redistribution.
