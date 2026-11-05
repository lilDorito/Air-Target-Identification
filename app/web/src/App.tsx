import { useEffect, useMemo, useState } from "react";
import { getMeta, predict, type Meta, type Observation, type Prediction, type Ranked } from "./api";

type FieldDef = { key: string; label: string; unit?: string; kind: "number" | "select"; hint?: string };

const FIELDS: Record<string, FieldDef> = {
  speed: { key: "speed", label: "Speed", unit: "km/h", kind: "number", hint: "e.g. 850" },
  altitude: { key: "altitude", label: "Altitude", unit: "m", kind: "number", hint: "e.g. 9000" },
  climb_rate: { key: "climb_rate", label: "Vertical rate", unit: "m/s", kind: "number", hint: "climb or descent" },
  turn_rate: { key: "turn_rate", label: "Turn rate", unit: "°/s", kind: "number", hint: "0 if flying straight" },
  rcs: { key: "rcs", label: "Radar cross-section", unit: "m²", kind: "number", hint: "e.g. 3" },
  engine_type: { key: "engine_type", label: "Engine type", kind: "select" },
  engine_count: { key: "engine_count", label: "Engines", kind: "number", hint: "count" },
  afterburner: { key: "afterburner", label: "Afterburner", kind: "select" },
  ir_category: { key: "ir_category", label: "Heat signature", kind: "select" },
  length_m: { key: "length_m", label: "Length", unit: "m", kind: "number" },
  wingspan_m: { key: "wingspan_m", label: "Wingspan", unit: "m", kind: "number" },
  mtow_kg: { key: "mtow_kg", label: "Take-off mass", unit: "kg", kind: "number" },
  first_flight: { key: "first_flight", label: "First flight", unit: "year", kind: "number" },
};

const RADAR = ["speed", "altitude", "climb_rate", "turn_rate", "rcs"];

const TYPE_WORD: Record<string, string> = {
  "Fighter-class fixed-wing": "fighter-class jet",
  "Heavy fixed-wing": "large aircraft",
  "Rotary-wing": "helicopter",
  UAV: "drone",
  "Missile/munition": "missile",
};

const ROLE_TYPE: Record<string, string> = {
  "Air superiority fighter": "Fighter-class fixed-wing", Interceptor: "Fighter-class fixed-wing",
  "Multirole fighter": "Fighter-class fixed-wing", "Strike fighter": "Fighter-class fixed-wing",
  "CAS/attack": "Fighter-class fixed-wing", "EW/SEAD": "Fighter-class fixed-wing",
  Reconnaissance: "Fighter-class fixed-wing", "Tactical bomber": "Fighter-class fixed-wing",
  "Trainer/light combat": "Fighter-class fixed-wing",
  "AEW&C": "Heavy fixed-wing", Bomber: "Heavy fixed-wing", "Maritime patrol": "Heavy fixed-wing",
  "Special mission (ISR/EW)": "Heavy fixed-wing", Tanker: "Heavy fixed-wing", Transport: "Heavy fixed-wing",
  Attack: "Rotary-wing", "Naval/ASW": "Rotary-wing", "Utility/transport": "Rotary-wing",
  CCA: "UAV", Decoy: "UAV", "MALE/HALE": "UAV", "One-way attack": "UAV", "Tactical recon": "UAV",
  "Anti-radiation": "Missile/munition", "Anti-ship": "Missile/munition", "Land-attack": "Missile/munition",
};

const roleKind = (role: string) => TYPE_WORD[ROLE_TYPE[role]] ?? "";

const SCENARIO_TEXT: Record<string, { name: string; about: string }> = {
  A_radar: { name: "Radar only", about: "Kinematics and radar cross-section from a single radar." },
  B_radar_sensors: { name: "Radar and sensors", about: "Adds engine and heat cues from IR or optical sensors." },
  C_all: { name: "Full description", about: "Adds size and mass. An upper bound, not a field scenario." },
};

const pct = (p: number) => (p >= 0.995 ? ">99%" : p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`);

function cleanObservation(values: Record<string, string>, inputs: string[]): Observation {
  const obs: Observation = {};
  for (const key of inputs) {
    const raw = values[key]?.trim();
    if (!raw) continue;
    obs[key] = FIELDS[key].kind === "number" ? Number(raw) : raw;
  }
  return obs;
}

function describe(obs: Observation): string {
  return RADAR.filter((k) => obs[k] !== undefined)
    .map((k) => `${FIELDS[k].label.toLowerCase()} ${obs[k]}${FIELDS[k].unit ? " " + FIELDS[k].unit : ""}`)
    .join(", ") || "sensor cues only";
}

function Bars({ items, title, source }: { items: Ranked[]; title: string; source?: string }) {
  return (
    <section className="ranking" aria-label={title}>
      <h3>
        {title}
        {source && <span className="source">from {source}</span>}
      </h3>
      <ol>
        {items.map((item, i) => (
          <li key={item.label} className={i === 0 ? "lead" : undefined}>
            <span className="name">
              <span className="label">{item.label}</span>
              {title === "Role" && roleKind(item.label) && <span className="kind">{roleKind(item.label)}</span>}
            </span>
            <span className="bar" aria-hidden="true">
              <span style={{ width: `${Math.max(item.probability * 100, 0.8)}%` }} />
            </span>
            <span className="value">{pct(item.probability)}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [scenarioId, setScenarioId] = useState("A_radar");
  const [values, setValues] = useState<Record<string, string>>({});
  const [track, setTrack] = useState<Observation[]>([]);
  const [countries, setCountries] = useState<string[]>([]);
  const [countryQuery, setCountryQuery] = useState("");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getMeta()
      .then((m) => {
        setMeta(m);
        if (m.scenarios.length && !m.scenarios.some((s) => s.id === "A_radar")) setScenarioId(m.scenarios[0].id);
      })
      .catch((e: Error) => setMetaError(e.message));
  }, []);

  const scenario = useMemo(() => meta?.scenarios.find((s) => s.id === scenarioId), [meta, scenarioId]);
  const inputs = scenario?.inputs ?? RADAR;
  const current = cleanObservation(values, inputs);
  const hasCurrent = Object.keys(current).length > 0;
  const full = meta?.mode === "full";
  // selected countries stay on top, the rest is narrowed by the search box
  const query = countryQuery.trim().toLowerCase();
  const shownCountries = [
    ...countries,
    ...(meta?.countries ?? []).filter((c) => !countries.includes(c) && (!query || c.toLowerCase().includes(query))),
  ];

  function changeScenario(id: string) {
    setScenarioId(id);
    setTrack([]);
    setResult(null);
    setError(null);
  }

  function addSnapshot() {
    if (!hasCurrent) return;
    setTrack((t) => [...t, current]);
    setValues((v) => Object.fromEntries(Object.entries(v).filter(([k]) => !RADAR.includes(k))));
  }

  async function identify() {
    const observations = hasCurrent ? [...track, current] : track;
    if (!observations.length) {
      setError("Enter at least one measured value, then identify.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      setResult(await predict(scenarioId, observations, countries));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (metaError)
    return (
      <main className="shell">
        {import.meta.env.DEV ? (
          <p className="notice warn">
            The identification service is not reachable ({metaError}). Start it with{" "}
            <code>uvicorn app.api.main:app</code> from the repository root, then reload this page.
          </p>
        ) : (
          <p className="notice warn">
            The identification service is not responding. It may be waking up, which takes about a minute.{" "}
            <button type="button" className="link" onClick={() => window.location.reload()}>
              Try again
            </button>
          </p>
        )}
      </main>
    );

  if (meta && meta.scenarios.length === 0)
    return (
      <main className="shell">
        <p className="notice warn">
          The service is running but has no models loaded yet. Build them with{" "}
          <code>python scripts/build_bundle.py</code>, or check <code>MODELS_URL</code> on the server.
        </p>
      </main>
    );

  const lead = result?.aircraft?.[0] ?? null;
  const leadRole = result?.role?.top[0] ?? null;

  return (
    <main className="shell">
      <header className="masthead">
        <h1>Air target identification</h1>
        <p>
          Describe what the sensors measured. The model ranks the aircraft, missiles and drones that fit, with
          calibrated probabilities.
        </p>
      </header>

      {meta && !full && (
        <p className="notice">
          Public demo: this version identifies role and type only. Naming the exact aircraft and limiting the
          answer to the countries that operate it are available in the private version.
        </p>
      )}

      <div className="layout">
        <section className="panel inputs" aria-label="Observation">
          <fieldset className="scenarios">
            <legend>What can you measure?</legend>
            {(meta?.scenarios ?? []).map((s) => (
              <label key={s.id} className={s.id === scenarioId ? "chosen" : undefined}>
                <input
                  type="radio"
                  name="scenario"
                  value={s.id}
                  checked={s.id === scenarioId}
                  onChange={() => changeScenario(s.id)}
                />
                <span className="scenario-name">{SCENARIO_TEXT[s.id]?.name ?? s.id}</span>
                <span className="scenario-about">{SCENARIO_TEXT[s.id]?.about}</span>
              </label>
            ))}
          </fieldset>

          <div className="fields">
            {inputs.map((key) => {
              const f = FIELDS[key];
              const id = `field-${key}`;
              return (
                <label key={key} htmlFor={id} className={RADAR.includes(key) ? "radar" : "cue"}>
                  <span className="field-label">
                    {f.label}
                    {f.unit && <span className="unit">{f.unit}</span>}
                  </span>
                  {f.kind === "number" ? (
                    <input
                      id={id}
                      type="number"
                      inputMode="decimal"
                      min={0}
                      step="any"
                      placeholder={f.hint ?? "not measured"}
                      value={values[key] ?? ""}
                      onChange={(e) => setValues({ ...values, [key]: e.target.value })}
                    />
                  ) : (
                    <select id={id} value={values[key] ?? ""} onChange={(e) => setValues({ ...values, [key]: e.target.value })}>
                      <option value="">not observed</option>
                      {(key === "engine_type" ? meta?.engine_types : key === "ir_category" ? meta?.ir_categories : ["yes", "no"])?.map(
                        (o) => (
                          <option key={o} value={o}>
                            {o}
                          </option>
                        ),
                      )}
                    </select>
                  )}
                </label>
              );
            })}
          </div>

          <div className="track">
            <h2>Track</h2>
            {track.length === 0 ? (
              <p className="muted">
                One reading is a single snapshot. Add several snapshots of the same contact to narrow the answer.
              </p>
            ) : (
              <ol>
                {track.map((obs, i) => (
                  <li key={i}>
                    <span>{describe(obs)}</span>
                    <button type="button" className="link" onClick={() => setTrack(track.filter((_, j) => j !== i))}>
                      Remove
                    </button>
                  </li>
                ))}
              </ol>
            )}
          </div>

          {full && meta!.countries.length > 0 && (
            <details className="origins" open={pickerOpen} onToggle={(e) => setPickerOpen(e.currentTarget.open)}>
              <summary>
                Limit to operators
                <span className="muted">{countries.length ? countries.join(", ") : "every country considered"}</span>
              </summary>
              <p className="muted">Keeps only the types these countries fly today, wherever they were built.</p>
              <input
                type="search"
                className="picker-search"
                placeholder="Find a country"
                aria-label="Find a country"
                value={countryQuery}
                onChange={(e) => setCountryQuery(e.target.value)}
              />
              <div className="chips" role="group" aria-label="Operators">
                {shownCountries.map((c) => {
                  const on = countries.includes(c);
                  return (
                    <button
                      type="button"
                      key={c}
                      aria-pressed={on}
                      className={on ? "chip on" : "chip"}
                      onClick={() => setCountries(on ? countries.filter((x) => x !== c) : [...countries, c])}
                    >
                      {c}
                    </button>
                  );
                })}
                {shownCountries.length === countries.length && countryQuery && (
                  <span className="muted">No operator matches “{countryQuery}”.</span>
                )}
              </div>
              {countries.length > 0 && (
                <button type="button" className="link" onClick={() => setCountries([])}>
                  Clear selection
                </button>
              )}
            </details>
          )}

          <div className="actions">
            <button type="button" className="primary" onClick={identify} disabled={busy}>
              {busy ? "Identifying…" : "Identify"}
            </button>
            <button type="button" onClick={addSnapshot} disabled={!hasCurrent}>
              Add snapshot to track
            </button>
            {track.length > 0 && (
              <button type="button" className="link" onClick={() => setTrack([])}>
                Clear track
              </button>
            )}
          </div>
          {error && (
            <p className="notice warn" role="alert">
              {error}
            </p>
          )}
        </section>

        <section className="panel results" aria-live="polite" aria-label="Identification">
          {!result ? (
            <div className="empty">
              <p>Enter what the sensors measured, then press Identify.</p>
              <p className="muted">Leave a field empty if it was not measured.</p>
            </div>
          ) : (
            <>
              <div className="readout" key={JSON.stringify(result)}>
                <p className="readout-context">
                  Most likely, from {result.snapshots} {result.snapshots === 1 ? "snapshot" : "snapshots"}
                  {result.filter ? `, operated by ${result.filter.join(", ")}` : ""}
                </p>
                {lead ? (
                  <>
                    <p className="readout-name">
                      {lead.label}
                      {lead.callsign && <span className="callsign"> {lead.callsign}</span>}
                    </p>
                    <p className="readout-detail">
                      {[lead.manufacturer, lead.country].filter(Boolean).join(" / ")}
                      <span>
                        {lead.role}, {TYPE_WORD[lead.type] ?? lead.type.toLowerCase()}
                      </span>
                    </p>
                    <p className="readout-p">{pct(lead.probability)}</p>
                  </>
                ) : (
                  leadRole && (
                    <>
                      <p className="readout-name">{leadRole.label}</p>
                      <p className="readout-detail">{roleKind(leadRole.label) || result.type?.top[0]?.label}</p>
                      <p className="readout-p">{pct(leadRole.probability)}</p>
                    </>
                  )
                )}
                {result.warnings?.map((w) => (<p key={w} className="notice warn">{w}</p>))}
              </div>

              {result.aircraft && (
                <section className="ranking" aria-label="Aircraft">
                  <h3>Candidates</h3>
                  <ol>
                    {result.aircraft.map((a, i) => (
                      <li key={a.label} className={i === 0 ? "lead" : undefined}>
                        <span className="name" title={[a.role, a.country].filter(Boolean).join(", ")}>
                          <span className="label">
                            {a.label}
                            {a.callsign && <span className="callsign"> {a.callsign}</span>}
                          </span>
                          <span className="kind">
                            {a.role}, {TYPE_WORD[a.type] ?? a.type.toLowerCase()}
                          </span>
                        </span>
                        <span className="bar" aria-hidden="true">
                          <span style={{ width: `${Math.max(a.probability * 100, 0.8)}%` }} />
                        </span>
                        <span className="value">{pct(a.probability)}</span>
                      </li>
                    ))}
                  </ol>
                </section>
              )}
              {result.role && <Bars title="Role" items={result.role.top} source={result.role.source} />}
              {result.type && <Bars title="Type" items={result.type.top} source={result.type.source} />}
            </>
          )}
          <p className="footnote">
            Trained on simulated observations drawn from published specifications, not on real sensor data.
          </p>
        </section>
      </div>
    </main>
  );
}
