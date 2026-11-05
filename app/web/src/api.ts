// Thin client for the FastAPI backend

const BASE = import.meta.env.VITE_API_URL ?? "";

export type Ranked = { label: string; probability: number };
export type AircraftCandidate = Ranked & {
  manufacturer: string | null;
  callsign: string | null;
  country: string | null;
  role: string;
  type: string;
};
export type Scenario = { id: string; inputs: string[]; targets: string[] };
export type Meta = {
  mode: "full" | "public";
  scenarios: Scenario[];
  engine_types: string[];
  ir_categories: string[];
  countries: string[];
  max_top_k: number;
};
export type Observation = Partial<Record<string, number | string>>;
export type Prediction = {
  scenario: string;
  snapshots: number;
  mode: "full" | "public";
  filter: string[] | null;
  aircraft: AircraftCandidate[] | null;
  role?: { source: string; top: Ranked[] };
  type?: { source: string; top: Ranked[] };
  warnings?: string[];
};

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const getMeta = () => call<Meta>("/api/meta");

export const predict = (scenario: string, observations: Observation[], countries: string[]) =>
  call<Prediction>("/api/predict", {
    method: "POST",
    body: JSON.stringify({
      scenario,
      observations,
      countries: countries.length ? countries : null,
      top_k: 5,
    }),
  });
