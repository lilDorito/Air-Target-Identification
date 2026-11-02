# Data Collection Notice

This document explains how the air-target dataset behind this project was built: where every value comes from, how much each kind of value can be trusted, and what the dataset deliberately does not contain.

**The dataset itself is not published in this repository.** See [Data handling and distribution](#7-data-handling-and-distribution).

---

## 1. Purpose and scope

The dataset supports a classification study: *how well can an air target be identified — as an individual aircraft type, and by role and coarse type — from the quantities a sensor could plausibly observe?*

| Item | Value |
|---|---|
| Entries | 535 aircraft, UAVs, missiles and guided munitions |
| Coarse labels (`type`) | Fighter-class fixed-wing, Heavy fixed-wing, Rotary-wing, UAV, Missile/munition |
| Fine labels (`role`) | 26 roles (e.g. air superiority, AEW&C, attack helicopter, one-way attack UAV, decoy, CCA, land-attack, anti-ship, anti-radiation) |
| Era | In service now, or at any time in roughly the last 60–70 years |
| Geographic focus | Worldwide, with emphasis on Europe, Ukraine, Israel, USA, USSR/Russia, China, Turkey, Australia |
| Reference date | November 2026 |

**Selection rules**

- A variant gets its own row when its role differs (F-15C / F-15E / F-15EX) or its airframe is substantially new (Geran-2 / Geran-5). Same-role sub-variants are collapsed into the most produced or used one.
- UAVs are fixed-wing only. FPV and multicopter drones are excluded.
- Missiles cover SRBM/MRBM/IRBM, cruise, glide and air-to-surface weapons. ICBMs/SLBMs, SAMs and air-to-air missiles are excluded.
- Roles follow the *designed* mission. For example, anti-ship missiles used against land targets stay `Anti-ship`, with a note.
- The two crewed fixed-wing types are split by airframe class as a sensor would see it, not by mission. This is why the SR-71 is `Fighter-class` and not `Heavy`.

## 2. Schema

| Group | Columns | Model input? |
|---|---|---|
| Identity | `name`, `manufacturer`, `callsign`, `family`, `country` | **Never.** Output strings for display. `family` is used for grouped validation and `country` for output filtering. |
| Labels | `type`, `role` (and `name` for the identification task) | Targets |
| Context | `ua_war`, `status`, `first_flight` | `first_flight` only in the full-feature scenario |
| Radar-observable | speed min/cruise/max, altitude min/max, max climb, max turn rate, RCS min/max | Yes |
| Other sensors | engine type, engine count, afterburner, IR category | Scenarios 2–3 |
| Descriptive | length, wingspan, MTOW | Scenario 3 only |
| Excluded | `flight_profile` | **Never**, it nearly states the coarse type (e.g. "rotary", "ballistic") |
| Provenance | `sources`, `estimated_fields`, `notes` | Never |

**Naming standard**

- `name` is the designation only: no manufacturer prefix and no factory code. The variant suffix is dropped when a family has only one row.
- `manufacturer` is one canonical company.
- `callsign` is the NATO reporting name for Soviet, Russian and Chinese systems, and otherwise the official popular name.

## 3. How the data was collected

Values come from three tiers of decreasing reliability. Every value records its tier, and a higher tier always overrides a lower one.

### Tier 1: Live-sourced research (133 rows)

All 115 rows used in the Russo-Ukrainian war were researched online one by one. So were 18 rows from the Tier 2 checks: 10 from the random verification sample and 8 from the full audit. Every row lists its URLs in `sources`, except Morok, for which no usable source exists. The sources, in order of preference:

1. **Primary / official:** Ukraine's Defence Intelligence (HUR) *War & Sanctions* portal; USAF and NAVAIR fact sheets; the Russian FSVTS export catalogue; national MoD pages.
2. **Manufacturer datasheets:** Baykar, AeroVironment, Insitu, WB Group, MBDA, Boeing, Lockheed Martin, Skyeton, Ukrspecsystems, Terminal Autonomy, Quantum Systems, Luch, TUSAS.
3. **Specialist analysts:** CSIS Missile Threat, FAS, Institute for Science and International Security, designation-systems.net.
4. **Reputable defence press:** for systems with no official data (Ukrainian missile-drones, new Geran variants), reports citing HUR or developers.
5. **Wikipedia, as a fallback only.** Marked "Wikipedia fallback" in `notes`. 32 researched rows rely on it, mostly Soviet-era aircraft and some Russian missiles.

Game wikis and forums (War Thunder and similar), auto-generated comparison sites and AI-generated encyclopedias were not used. `notes` records when a figure belongs to a related variant, for example "HUR figures are for the 9M727".

### Tier 2: Reference figures from AI recall (402 rows)

For the remaining rows, standard published specifications were filled from an AI assistant's general reference knowledge, not fetched live. These rows are labelled in `sources` as *"AI-recalled reference figures (unverified)"*. Originally there were 420 such rows; 18 were later moved to Tier 1 after checking, leaving 402.

Tier 2 was fact-checked in two ways.

**Check 1: random sample.** A random sample (Python `random.seed(42)`, 14 rows drawn; the 9 with checkable public data, plus one sibling variant) was checked against live sources:

| Result | Count |
|---|---|
| Figures checked | 42 |
| Within 5% of the source | 35 (83%) |
| Off by more than 5% | 7 |

The misses:

| Row | Field | Recalled | Source | Error |
|---|---|---|---|---|
| F-100 | length / MTOW / max speed / ceiling | 15.09 m / 15,800 kg / 1,390 km/h / 15,000 m | 16.51 m / 17,258 kg / 1,491 km/h / 16,764 m | 7–11% low |
| Aksungur | ceiling | 7,620 m | 12,192 m | 37% low |
| Aksungur | length | 11.6 m | 12.5 m | 7% low |
| Yak-28R | MTOW | 20,000 kg | 17,250 kg | 16% high |
| Yak-28B (sibling) | MTOW | 19,500 kg | 16,000 kg | 22% high |

All sampled rows were corrected and moved to Tier 1.

**Check 2: full audit of all recalled rows.** Every row was inspected in three passes:

1. **Automatic plausibility checks:** speed order (min ≤ cruise ≤ max), length vs mass (cube law), span vs length, engine vs speed (e.g. afterburner on a slow aircraft, piston engine above 700 km/h), helicopter ceilings, climb vs speed, and first-flight years. 9 rows were flagged. 7 turned out to be real design quirks (F-104's short wings, the slow F-89 with afterburner, the long C-5 and EMB-145, the V-22's ceiling, the RQ-170 flying wing). 2 were errors (Mi-26 length, MALD wingspan).
2. **Manual read of every row** against known reference figures. This found 17 suspicious rows: copy errors, variant mix-ups and inconsistent conventions.
3. **Live check of the high-impact suspects.** Six were checked against sources: 4 confirmed as errors, and 2 recalled values turned out correct (KF-21 speed, XQ-58 size).

| Row | Problem | Fix | Basis |
|---|---|---|---|
| Alpha Jet | Length 13.23 m (copied from AMX) | 11.75 m | MILAVIA (live) |
| ADM-160 MALD | Span 0.65 m, ceiling 9,000 m | 1.71 m, 12,200 m | designation-systems.net (live) |
| F-8 | Length/span slightly off; max speed 1,800 km/h and MTOW 13,000 kg too low | 16.84 / 10.87 m (live); 1,975 km/h, 15,420 kg | Vought (live) + recall correction |
| Hawk | MTOW 9,100 kg belonged to the Hawk 100, not the T1 | 7,750 kg; length/span from FAS | FAS (live) + recall correction |
| Yak-28P / Yak-28PP | Wrong family length/span | 21.36 / 12.06 m | GlobalSecurity (live) |
| VC10 K3 | Length of the standard VC10; the K3 is a Super VC10 | 52.32 m | Recall correction |
| E-8C | Engine typed as turbojet; it uses turbofans | turbofan | Recall correction |
| Mi-26, AH-64, AH-2, PAH-1, SA342, MH-60L, MH-60R | Overall length including rotors, against the helicopter convention (fuselage length) | Fuselage lengths | Recall correction |

**Known remaining issues in Tier 2:**
- AH-1Z, T129, LCH, ALH and ALH WSI may still use overall length with rotors (not confirmed).
- Yak-28P/PP take-off weight is probably about 15% high.
- Corrections marked "recall correction" were not checked live.

**Overall:** most Tier 2 figures are close to published values, but about 1 in 6 checked figures was off by more than 5%, sometimes by 10–40%. The audit removed the clearest errors without guaranteeing the rest.

### Tier 3: Rule-based estimates (gap filling)

Any cell still empty after Tiers 1–2 is filled by a fixed, documented rule. Every such cell is listed in the row's `estimated_fields`.

**Design principle:** the rules depend only on physics and size (engine type, afterburner, mass, flight profile), **never on the role label**, so the estimates cannot encode the answer.

| Quantity | Rule |
|---|---|
| Physical class | From flight profile, engine, afterburner and mass: rotary / ballistic / glide / cruise missile / heavy jet (≥60 t) / fast jet (afterburner) / subsonic jet / heavy prop (≥15 t) / light |
| Size | length = 0.65 × MTOW^(1/3) (and inverse). Span = 0.9 × length; cruise missiles and glide weapons 0.4, ballistic 0.15, rotor 0.85 |
| Speeds | max = cruise / 0.85; cruise = 0.85 × max, capped at 900 km/h for jets; min = 0.25 × cruise (≥180 km/h) for jets, 0.35 × cruise for props, 0 for rotary |
| Altitude | Ceiling and minimum altitude by physical class (e.g. rotary 5–4,500 m, fast jet 50–15,000 m) |
| Climb | By class (e.g. fast jet 230 m/s, heavy jet 15 m/s, rotary 10 m/s) |
| Turn rate | 9.81·√(g²−1)/V at V = cruise speed capped at 900 km/h. g is the published limit where known, else a class default, lift-limited by stall speed |
| RCS | 0.03 × length × span m² (× 0.05 for low-observable designs), range × 0.5 to × 2. Exception: Parodiya, a decoy documented to carry radar reflectors that mimic a Shahed |
| IR category | Rule from engine type, afterburner and engine count only |

A small number of OSINT-typical values (for example glide-bomb release speeds and first-flight years of new Ukrainian drones) were entered by hand where no rule applied. These are also listed in `estimated_fields`.

**Left empty on purpose (no credible public figure):** B-21 size; dimensions of Italmas, UJ-25, FP-1, Morok, Peklo, Ruta and Shark; Morok's engine; Oreshnik's stage count; first flights of Granat-4, Ababil-3 and Hrim-2.

**In total:** 99.5% of feature cells are filled, and about 36% of the filled cells are Tier 3 estimates. Most of these are RCS, turn rate, climb and minimum altitude.

## 4. Collection principles

1. **Sourced beats recalled, and recalled beats estimated.**
2. **Every value says where it came from.** There are no silent fills.
3. **An empty cell beats an invented one** when nothing defensible exists.
4. **Estimates never use the label.**
5. **Uncertainty is measured, not hidden.** The recall error rate comes from a random, reproducible sample.

## 5. Known limitations and biases

- **Spec sheets are not observations.** Values are published maxima and nominal figures. A real sensor sees one speed at one moment. Models are trained on observations sampled from these ranges, which is a simulation assumption.
- **RCS and turn rate are mostly estimated.** Public RCS figures barely exist, and published g-limits cover few types. Their influence is measured with an ablation (with and without estimated cells).
- **Estimates follow physical class, which correlates with type.** For example, every helicopter gets minimum speed 0 by rule. This partly encodes the coarse label in estimated cells.
- **Recall errors.** About 1 in 6 checked Tier 2 figures was off by more than 5%. The audit fixed the clearest ones; others likely remain.
- **Mixed conventions.** "Maximum speed" is sometimes quoted at altitude and sometimes at sea level (a Mach number converts differently). A few helicopter lengths may include rotors.
- **Near-identical variants.** Licensed copies and close derivatives (e.g. Su-27S / J-11B, Su-30SM / J-16) have nearly identical specifications and cannot be separated from kinematics alone.
- **Coverage bias.** Western and Soviet/Russian types are better documented than Chinese, Iranian or new Ukrainian systems. Secretive systems carry more estimates.
- **Labels involve judgment.** Multirole aircraft and missile roles are assigned by designed mission, a documented rule rather than an objective fact.
- **Snapshot in time.** New variants (e.g. Geran-4/5) are described from 2025–26 intelligence reporting that is still evolving.

## 6. Tools

Research was done with AI-assisted web search and an AI assistant (Claude, Anthropic) for spreadsheet generation, the Tier 2 reference figures and the audit passes. Every step: roster, taxonomy, sources, recalled values, verification, audit and estimation rules - was reviewed and approved by the authors.

## 7. Data handling and distribution

The spreadsheet and any derived tables (CSV, sampled observations) are **kept out of this repository** (excluded via `.gitignore`). All values come from public sources or general reference knowledge, but we choose not to redistribute a compiled targeting-style table.

The repository contains documentation, code and trained models only. Note that a model identifying **individual aircraft** effectively encodes the table. Publishing it partly publishes the data, which should be considered before releasing it.

## 8. Updating the data

Corrections go into the source-data scripts kept with the dataset: researched values with a source URL, or recalled values. The workbook is regenerated from them. A sourced value automatically overrides a recalled or estimated one, so the provenance stays consistent.
