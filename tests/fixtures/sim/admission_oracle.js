// Pure extract of labs/sim.js simulateAdmission + p99. No DOM. Prints JSON.
// Scenario numbers are copied from labs/scenarios/today.json.
"use strict";

function p99(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.ceil(0.99 * sorted.length) - 1);
  return sorted[index];
}

function simulateAdmission(lab, repair) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  let cap = lab.kv_slots;
  let engines = lab.kv_slots;
  let enforce = lab.broken !== "admission";
  let aMul = 1;
  let bMul = 1;
  if (repair === "admission") enforce = true;
  if (repair === "prefill") aMul = 0.5;
  if (repair === "decode") bMul = 0.5;
  if (repair === "kv") {
    cap = lab.kv_slots * 2;
    engines = cap;
    enforce = true;
  }

  const jobs = lab.arrivals.map((row) => ({
    id: row.request_id,
    arrival: row.arrival_time_ms,
    prompt: row.prompt_tokens,
    output: row.output_tokens,
    phase: "pending",
    tokensLeft: row.output_tokens,
    ttft: null,
    itls: [],
  }));

  let now = 0;
  let free = engines;
  let occupied = 0;
  const waiting = [];
  const decoding = [];
  let gen = 0;
  const events = jobs.map((job) => ({ t: job.arrival, type: "arrive", job }));

  function scheduleDecode() {
    if (!decoding.length) return;
    gen += 1;
    const dt = bMs * bMul * decoding.length;
    events.push({ t: now + dt, type: "decode", gen, dt });
  }

  function pump() {
    while (waiting.length && free > 0) {
      const job = waiting.shift();
      job.phase = "prefill";
      free -= 1;
      events.push({ t: now + job.prompt * aMs * aMul, type: "prefill", job });
    }
  }

  while (events.length) {
    events.sort((left, right) => left.t - right.t);
    const event = events.shift();
    now = event.t;
    if (event.type === "arrive") {
      if (enforce && occupied >= cap) {
        event.job.phase = "rejected";
      } else {
        occupied += 1;
        event.job.phase = "waiting";
        waiting.push(event.job);
        pump();
      }
    } else if (event.type === "prefill") {
      event.job.phase = "decode";
      free += 1;
      decoding.push(event.job);
      pump();
      scheduleDecode();
    } else if (event.type === "decode" && event.gen === gen) {
      decoding.forEach((job) => {
        job.itls.push(event.dt);
        if (job.ttft === null) job.ttft = now - job.arrival;
        job.tokensLeft -= 1;
      });
      for (let i = decoding.length - 1; i >= 0; i -= 1) {
        if (decoding[i].tokensLeft > 0) continue;
        decoding[i].phase = "done";
        decoding.splice(i, 1);
        occupied -= 1;
      }
      scheduleDecode();
    }
  }

  const done = jobs.filter((job) => job.phase === "done");
  const admitted = done.length;
  const good = done.filter(
    (job) => job.ttft <= lab.slo.ttft_ms && p99(job.itls) <= lab.slo.itl_ms,
  ).length;
  return {
    ttft_p99_ms: p99(done.map((job) => job.ttft)),
    itl_p99_ms: p99(done.map((job) => p99(job.itls))),
    goodput: admitted ? good / admitted : 0,
    rejects: jobs.filter((job) => job.phase === "rejected").length,
    admitted,
  };
}

const lab = {
  prefill_us_per_token: 20,
  decode_us_per_token: 2000,
  kv_slots: 8,
  broken: "admission",
  slo: { ttft_ms: 500, itl_ms: 20 },
  arrivals: [
    { request_id: "burst-a-01", arrival_time_ms: 0, prompt_tokens: 20000, output_tokens: 512 },
    { request_id: "burst-a-02", arrival_time_ms: 40, prompt_tokens: 19800, output_tokens: 384 },
    { request_id: "burst-a-03", arrival_time_ms: 80, prompt_tokens: 19500, output_tokens: 256 },
    { request_id: "burst-a-04", arrival_time_ms: 120, prompt_tokens: 18000, output_tokens: 512 },
    { request_id: "burst-a-05", arrival_time_ms: 160, prompt_tokens: 17000, output_tokens: 128 },
    { request_id: "burst-a-06", arrival_time_ms: 200, prompt_tokens: 16000, output_tokens: 256 },
    { request_id: "burst-b-01", arrival_time_ms: 5000, prompt_tokens: 20000, output_tokens: 512 },
    { request_id: "burst-b-02", arrival_time_ms: 5050, prompt_tokens: 19000, output_tokens: 384 },
    { request_id: "burst-b-03", arrival_time_ms: 5100, prompt_tokens: 18500, output_tokens: 256 },
    { request_id: "burst-b-04", arrival_time_ms: 5150, prompt_tokens: 17500, output_tokens: 128 },
    { request_id: "steady-01", arrival_time_ms: 10000, prompt_tokens: 15000, output_tokens: 256 },
    { request_id: "steady-02", arrival_time_ms: 15000, prompt_tokens: 20000, output_tokens: 512 },
  ],
};

const packed = {
  ...lab,
  arrivals: lab.arrivals.map((row, index) => ({ ...row, arrival_time_ms: index * 20 })),
};

function run(scenario) {
  const out = {};
  for (const repair of [null, "admission", "prefill", "decode", "kv"]) {
    out[repair === null ? "null" : repair] = simulateAdmission(scenario, repair);
  }
  return out;
}

process.stdout.write(`${JSON.stringify({ today: run(lab), packed: run(packed) })}\n`);
