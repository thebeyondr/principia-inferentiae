// Metric extract of labs/day3/sim.js, labs/day4/sim.js, and the non-admission
// kernels in labs/sim.js. No DOM. Prints one JSON object per case.
"use strict";

function p99(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil(0.99 * sorted.length) - 1)];
}

function fillItl(job, gap, n) {
  job.itls = Array.from({ length: n }, () => gap);
  job.phase = "done";
}

function act(lab, choice) {
  if (!choice) return null;
  return (lab.effects && lab.effects[choice]) || choice;
}

function packDay(lab, jobs) {
  const done = jobs.filter((j) => j.phase === "done");
  const good = done.filter((j) => (
    (j.ttft == null || j.ttft <= lab.slo.ttft_ms) && p99(j.itls) <= lab.slo.itl_ms
  )).length;
  return {
    ttft_p99_ms: p99(done.map((j) => j.ttft).filter((v) => v != null)),
    itl_p99_ms: p99(done.map((j) => p99(j.itls))),
    goodput: done.length ? good / done.length : 0,
    rejects: jobs.filter((j) => j.phase === "rejected").length,
    admitted: done.length,
  };
}

function packLabs(lab, jobs) {
  const done = jobs.filter((j) => j.phase === "done");
  const good = done.filter((j) => j.ttft <= lab.slo.ttft_ms && p99(j.itls) <= lab.slo.itl_ms).length;
  return {
    ttft_p99_ms: p99(done.map((j) => j.ttft)),
    itl_p99_ms: p99(done.map((j) => p99(j.itls))),
    goodput: done.length ? good / done.length : 0,
    rejects: jobs.filter((j) => j.phase === "rejected").length,
    admitted: done.length,
  };
}

function simulateAdmit(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  const capacity = lab.kv_slots;
  let admitCap = Infinity;
  let engines = 1;
  if (choice === "a") admitCap = capacity;
  if (choice === "b") engines = 3;
  if (choice === "d") admitCap = 8;
  const jobs = [];
  let live = 0;
  lab.arrivals.forEach((row) => {
    const output = choice === "c" ? Math.ceil(row.output_tokens / 2) : row.output_tokens;
    if (live >= admitCap) {
      jobs.push({ phase: "rejected", ttft: null, itls: [] });
      return;
    }
    live += 1;
    const gap = bMs * Math.ceil(live / engines);
    const job = { phase: "pending", ttft: row.prompt_tokens * aMs + gap, itls: [] };
    fillItl(job, gap, output);
    jobs.push(job);
  });
  return packDay(lab, jobs);
}

function simulateQueue(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  const move = act(lab, choice);
  const fasterPrefill = move === "faster_prefill";
  const fasterDecode = move === "faster_decode";
  const moreEngines = move === "add_engines";
  const hideQueue = move === "faster_prefill" || move === "faster_decode";
  let serverFree = 0;
  const jobs = lab.arrivals.map((row, i) => {
    const promptMul = fasterPrefill ? 0.5 : 1;
    const gapMul = fasterDecode ? 0.5 : 1;
    const engines = moreEngines ? 4 : 1;
    const prefill = row.prompt_tokens * aMs * promptMul;
    const gap = bMs * gapMul;
    const start = moreEngines
      ? row.arrival_time_ms + (i % engines) * 5
      : Math.max(row.arrival_time_ms, serverFree);
    const queueWait = start - row.arrival_time_ms;
    const inferTtft = prefill + gap;
    const e2e = queueWait + inferTtft;
    if (!moreEngines) serverFree = start + prefill + row.output_tokens * gap;
    const job = { phase: "done", ttft: hideQueue ? inferTtft : e2e, itls: [] };
    fillItl(job, gap, row.output_tokens);
    return job;
  });
  return packDay(lab, jobs);
}

function simulateChunk(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  const move = act(lab, choice);
  const chunk = move === "chunk";
  const chunkSize = lab.chunk_tokens || 2048;
  const long = lab.arrivals[0];
  const shorts = lab.arrivals.slice(1);
  const jobs = [];
  if (chunk) {
    let t = 0;
    let remaining = long.prompt_tokens;
    while (remaining > 0) {
      const piece = Math.min(chunkSize, remaining);
      t += piece * aMs;
      remaining -= piece;
      shorts.forEach(() => { t += bMs; });
    }
    const longJob = { phase: "done", ttft: t, itls: [] };
    fillItl(longJob, bMs, long.output_tokens);
    jobs.push(longJob);
    shorts.forEach((row) => {
      const job = { phase: "done", ttft: row.prompt_tokens * aMs + bMs, itls: [] };
      fillItl(job, bMs, row.output_tokens);
      jobs.push(job);
    });
  } else {
    const preMul = move === "faster_prefill" ? 0.5 : 1;
    const decMul = move === "faster_decode" ? 0.5 : 1;
    const longPrefill = long.prompt_tokens * aMs * preMul;
    const longJob = { phase: "done", ttft: longPrefill + bMs * decMul, itls: [] };
    fillItl(longJob, bMs * decMul, long.output_tokens);
    jobs.push(longJob);
    shorts.forEach((row) => {
      if (move === "reject_shorts") {
        jobs.push({ phase: "rejected", ttft: null, itls: [] });
        return;
      }
      const job = {
        phase: "done",
        ttft: longPrefill + row.prompt_tokens * aMs * preMul + bMs * decMul,
        itls: [],
      };
      fillItl(job, bMs * decMul * (1 + shorts.length), row.output_tokens);
      jobs.push(job);
    });
  }
  return packDay(lab, jobs);
}

function simulateRetry(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  const move = act(lab, choice);
  let multiplier = lab.retry_mul || 3;
  let timeoutMs = lab.timeout_ms || 200;
  let admitCap = Infinity;
  if (move === "stop_retries") multiplier = 1;
  if (move === "cut_timeout") timeoutMs = 50;
  if (move === "admit_cap") admitCap = lab.kv_slots;
  if (move === "half_retries") multiplier = 2;
  const traffic = [];
  lab.arrivals.forEach((row) => {
    for (let r = 0; r < multiplier; r += 1) {
      traffic.push({ ...row, arrival_time_ms: row.arrival_time_ms + r * 30 });
    }
  });
  const jobs = [];
  let live = 0;
  traffic.forEach((row) => {
    if (live >= admitCap) {
      jobs.push({ phase: "rejected", ttft: null, itls: [] });
      return;
    }
    live += 1;
    const gap = bMs * live;
    const ttft = row.prompt_tokens * aMs + gap;
    if (ttft > timeoutMs && move !== "stop_retries") {
      jobs.push({ phase: "rejected", ttft, itls: [] });
      live -= 1;
      return;
    }
    const job = { phase: "pending", ttft, itls: [] };
    fillItl(job, gap, row.output_tokens);
    jobs.push(job);
  });
  return packDay(lab, jobs);
}

function simulateGoodput(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  const move = act(lab, choice);
  let batch = lab.arrivals.length;
  let step = bMs;
  if (move === "cap4") batch = lab.kv_slots;
  if (move === "pack12") batch = lab.arrivals.length;
  if (move === "faster_decode_pack") step = bMs * 0.5;
  if (move === "admit6") batch = Math.ceil(lab.arrivals.length / 2);
  const gap = step * batch;
  const jobs = lab.arrivals.map((row, i) => {
    if (move === "cap4" && i >= lab.kv_slots) return { phase: "rejected", ttft: null, itls: [] };
    if (move === "admit6" && i >= batch) return { phase: "rejected", ttft: null, itls: [] };
    const job = { phase: "done", ttft: row.prompt_tokens * aMs + gap, itls: [] };
    fillItl(job, gap, row.output_tokens);
    return job;
  });
  return packDay(lab, jobs);
}

function simulatePrefix(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  const move = act(lab, choice);
  const prefixTok = lab.prefix_tokens || 4000;
  const jobs = lab.arrivals.map((row, i) => {
    let prompt = row.prompt_tokens;
    if (move === "shorter_prompts") prompt = Math.ceil(prompt * 0.25);
    let prefillTok = prompt;
    if (move === "enable_prefix" && i > 0) prefillTok = Math.max(0, prompt - prefixTok);
    if (move === "more_engines" && i > 0) prefillTok = prompt;
    const gap = bMs * (move === "faster_decode" ? 0.5 : 1) * Math.min(i + 1, 4);
    const job = { phase: "done", ttft: prefillTok * aMs + gap, itls: [] };
    fillItl(job, gap, row.output_tokens);
    return job;
  });
  return packDay(lab, jobs);
}

function simulatePriority(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  const move = act(lab, choice);
  const preempt = move === "preempt_batch";
  const admitCap = move === "admit_fewer" ? 2 : Infinity;
  const fastDec = move === "faster_decode";
  const jobs = [];
  let live = 0;
  lab.arrivals.forEach((row) => {
    const interactive = row.priority === "interactive";
    if (live >= admitCap) {
      jobs.push({ phase: "rejected", ttft: null, itls: [] });
      return;
    }
    live += 1;
    let concurrent = live;
    if (preempt && interactive) concurrent = Math.max(1, live - 1);
    if (move === "more_kv") concurrent = live;
    const steal = !preempt && !interactive ? 1 : (!preempt && interactive ? live : concurrent);
    const realGap = bMs * (fastDec ? 0.5 : 1) * (preempt && interactive ? 1 : steal);
    const job = { phase: "pending", ttft: row.prompt_tokens * aMs + realGap, itls: [] };
    fillItl(job, realGap, row.output_tokens);
    jobs.push(job);
  });
  return packDay(lab, jobs);
}

function simulateSticky(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  const move = act(lab, choice);
  const prefixTok = lab.prefix_tokens || 3000;
  const seenUser = {};
  const jobs = lab.arrivals.map((row, i) => {
    let prefillTok = row.prompt_tokens;
    if (move === "sticky_route" && i > 0) prefillTok = Math.max(0, row.prompt_tokens - prefixTok);
    else if (move === "pin_user") {
      if (seenUser[row.user_id]) prefillTok = Math.max(0, row.prompt_tokens - prefixTok);
      seenUser[row.user_id] = true;
    } else if (move === "faster_prefill") prefillTok = Math.ceil(row.prompt_tokens * 0.5);
    const job = { phase: "done", ttft: prefillTok * aMs + bMs, itls: [] };
    fillItl(job, bMs, row.output_tokens);
    return job;
  });
  return packDay(lab, jobs);
}

function simulateHog(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  const move = act(lab, choice);
  const capOut = move === "cap_output";
  const preempt = move === "preempt_long";
  const fastDec = move === "faster_decode";
  const jobs = lab.arrivals.map((row) => {
    let out = row.output_tokens;
    if (capOut && out > (lab.max_output || 128)) out = lab.max_output || 128;
    if (move === "reject_shorts" && row.output_tokens <= 64) {
      return { phase: "rejected", ttft: null, itls: [] };
    }
    let concurrent = lab.arrivals.length;
    if (preempt && row.output_tokens <= 64) concurrent = 2;
    if (capOut) concurrent = Math.min(lab.arrivals.length, 4);
    if (move === "more_capacity") concurrent = 2;
    const gap = bMs * (fastDec ? 0.5 : 1) * concurrent;
    const job = { phase: "done", ttft: row.prompt_tokens * aMs + gap, itls: [] };
    fillItl(job, gap, out);
    return job;
  });
  return packDay(lab, jobs);
}

function simulateFrag(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  const move = act(lab, choice);
  const paged = move === "paged_blocks";
  const buyMem = move === "buy_memory";
  const lowerBatch = move === "lower_batch";
  const slots = lab.kv_slots;
  const jobs = [];
  let live = 0;
  const fragmented = !paged && !buyMem;
  lab.arrivals.forEach((row) => {
    const need = 1;
    const freeSlots = buyMem ? slots * 2 : slots;
    const usable = paged
      ? freeSlots - live
      : (fragmented && live >= Math.floor(slots * 0.5) ? 0 : freeSlots - live);
    if (usable < need || (lowerBatch && live >= 2)) {
      if (lowerBatch && live >= 2) {
        jobs.push({ phase: "rejected", ttft: null, itls: [] });
        return;
      }
      if (!paged && !buyMem && live >= Math.floor(slots * 0.5)) {
        jobs.push({ phase: "rejected", ttft: null, itls: [] });
        return;
      }
    }
    if (live >= (buyMem ? slots * 2 : slots)) {
      jobs.push({ phase: "rejected", ttft: null, itls: [] });
      return;
    }
    live += 1;
    const gap = bMs * live;
    const job = { phase: "pending", ttft: row.prompt_tokens * aMs + gap, itls: [] };
    fillItl(job, gap, row.output_tokens);
    jobs.push(job);
  });
  return packDay(lab, jobs);
}

function simulateSpec(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const baseGap = lab.decode_us_per_token / 1000;
  const move = act(lab, choice);
  let k = lab.draft_k || 4;
  let alpha = lab.accept || 0.2;
  if (move === "raise_accept") alpha = 0.8;
  if (move === "bigger_k") k = 16;
  if (move === "faster_draft") alpha = 0.2;
  if (move === "disable_spec") { k = 1; alpha = 1; }
  const draftCost = move === "faster_draft" ? baseGap * 0.25 : baseGap * 0.5;
  const gained = 1 + alpha * (k - 1);
  const gap = move === "disable_spec" ? baseGap : (draftCost * k + baseGap) / Math.max(gained, 0.01);
  const row = lab.arrivals[0];
  const job = { phase: "done", ttft: row.prompt_tokens * aMs + gap, itls: [] };
  fillItl(job, gap, row.output_tokens);
  return packDay(lab, [job]);
}

function simulateEvict(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  const move = act(lab, choice);
  const cacheBlocks = lab.cache_blocks || 4;
  let hitRate = 0;
  if (move === "grow_cache") hitRate = 0.85;
  else if (move === "pin_hot") hitRate = 0.75;
  else if (move === "faster_prefill") hitRate = 0.1;
  else if (move === "more_replicas") hitRate = 0.15;
  else hitRate = cacheBlocks / Math.max(lab.arrivals.length, 1);
  const prefixTok = lab.prefix_tokens || 2000;
  const jobs = lab.arrivals.map((row, i) => {
    const hitsAllowed = Math.floor((lab.arrivals.length - 1) * hitRate);
    const isHit = i > 0 && i <= hitsAllowed;
    const prefillTok = isHit
      ? Math.max(0, row.prompt_tokens - prefixTok)
      : (move === "faster_prefill" ? Math.ceil(row.prompt_tokens * 0.5) : row.prompt_tokens);
    const job = { phase: "done", ttft: prefillTok * aMs + bMs, itls: [] };
    fillItl(job, bMs, row.output_tokens);
    return job;
  });
  return packDay(lab, jobs);
}

function simulateLora(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  const move = act(lab, choice);
  const swapMs = lab.swap_ms || 40;
  if (move === "queue_by_adapter") {
    const sorted = [...lab.arrivals].sort((a, b) => a.adapter - b.adapter);
    return packDay(lab, sorted.map((row, i) => {
      const first = sorted.findIndex((r) => r.adapter === row.adapter) === i;
      const swap = first ? swapMs : 0;
      const job = { phase: "done", ttft: swap + row.prompt_tokens * aMs + bMs, itls: [] };
      fillItl(job, bMs, row.output_tokens);
      return job;
    }));
  }
  const loaded = new Set();
  const adaptersResident = move === "pin_adapters" ? (lab.unique_adapters || 4) : 1;
  const jobs = lab.arrivals.map((row, i) => {
    let swap = 0;
    if (move === "merge_one") swap = 0;
    else if (move === "pin_adapters") {
      if (!loaded.has(row.adapter) && loaded.size >= adaptersResident) {
        loaded.delete(loaded.values().next().value);
      }
      if (!loaded.has(row.adapter)) {
        swap = swapMs;
        loaded.add(row.adapter);
      }
    } else {
      const prev = i === 0 ? null : lab.arrivals[i - 1].adapter;
      if (row.adapter !== prev) swap = swapMs;
    }
    const gap = bMs * (move === "faster_decode" ? 0.5 : 1);
    const job = { phase: "done", ttft: swap + row.prompt_tokens * aMs + gap, itls: [] };
    fillItl(job, gap, row.output_tokens);
    return job;
  });
  return packDay(lab, jobs);
}

function simulateScale(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  const move = act(lab, choice);
  let engines = 1;
  const warmDelay = lab.warm_ms || 3000;
  if (move === "prewarm") engines = 3;
  if (move === "faster_scale") engines = 3;
  if (move === "admit_shed") engines = 1;
  if (move === "bigger_machine") engines = 1;
  const readyAt = move === "prewarm" ? 0 : (move === "faster_scale" ? 500 : warmDelay);
  const jobs = lab.arrivals.map((row, i) => {
    if (move === "admit_shed" && i >= lab.kv_slots) return { phase: "rejected", ttft: null, itls: [] };
    const available = (move === "prewarm" || (move === "faster_scale" && row.arrival_time_ms >= readyAt))
      ? engines
      : 1;
    const conc = move === "bigger_machine"
      ? Math.min(lab.arrivals.length, 8)
      : Math.ceil((i + 1) / Math.max(available, 1));
    const queue = move === "prewarm" ? 0 : Math.max(0, (i - available + 1) * 40);
    const gap = bMs * Math.min(conc, move === "bigger_machine" ? 8 : 4);
    const job = { phase: "done", ttft: queue + row.prompt_tokens * aMs + gap, itls: [] };
    fillItl(job, gap, row.output_tokens);
    return job;
  });
  return packDay(lab, jobs);
}

function simulateBatch(lab, choice) {
  const aMs = lab.prefill_us_per_token / 1000;
  const bMs = lab.decode_us_per_token / 1000;
  const move = act(lab, choice);
  const continuous = move === "continuous";
  const smallerStatic = move === "smaller_static";
  const fillTarget = move === "bigger_static" ? 8 : (smallerStatic ? 2 : 4);
  const jobs = lab.arrivals.map((row, i) => {
    let start;
    if (continuous) start = row.arrival_time_ms;
    else {
      const idxInBatch = i % fillTarget;
      const batchReady = i - idxInBatch + fillTarget - 1;
      start = lab.arrivals[Math.min(batchReady, lab.arrivals.length - 1)].arrival_time_ms;
    }
    const wait = start - row.arrival_time_ms;
    const gap = bMs * (move === "faster_decode" ? 0.5 : 1) * (continuous ? Math.min(i + 1, 4) : fillTarget);
    const job = { phase: "done", ttft: wait + row.prompt_tokens * aMs + gap, itls: [] };
    fillItl(job, gap, row.output_tokens);
    return job;
  });
  return packDay(lab, jobs);
}

function simulateLabsPrefix(lab, repair) {
  const aMs = (lab.prefill_us_per_token / 1000) * (repair === "prefill" ? 0.5 : 1);
  const gap = (lab.decode_us_per_token / 1000) * lab.arrivals.length * (repair === "decode" ? 0.5 : 1);
  const cache = repair === "prefix";
  const jobs = lab.arrivals.map((row) => {
    const tokens = cache ? row.prompt_tokens - row.shared_prefix_tokens : row.prompt_tokens;
    const job = { phase: "done", ttft: tokens * aMs + gap, itls: [] };
    fillItl(job, gap, row.output_tokens);
    return job;
  });
  return packLabs(lab, jobs);
}

function simulateLabsSpec(lab, repair) {
  const row = lab.arrivals[0];
  const prefill = row.prompt_tokens * (lab.prefill_us_per_token / 1000);
  if (repair === "off") {
    const targetMs = lab.verify_us / 1000;
    const job = { phase: "done", ttft: prefill + targetMs, itls: [] };
    fillItl(job, targetMs, row.output_tokens);
    return packLabs(lab, [job]);
  }
  let draft = lab.draft_len;
  let draftUs = lab.draft_us_per_token;
  if (repair === "match") draft = lab.accept_len;
  if (repair === "faster") draftUs *= 0.5;
  const roundMs = (draft * draftUs + lab.verify_us) / 1000;
  const committed = Math.min(lab.accept_len, draft);
  const gap = roundMs / committed;
  const job = { phase: "done", ttft: prefill + roundMs, itls: [] };
  fillItl(job, gap, row.output_tokens);
  return packLabs(lab, [job]);
}

function simulateDevice(lab, repair) {
  const row = lab.arrivals[0];
  let compute = lab.compute_ms;
  let link = lab.link_ms;
  if (repair === "device") { compute = lab.local_ms; link = 0; }
  if (repair === "decode") compute *= 0.5;
  if (repair === "link") link *= 0.5;
  const gap = compute + link;
  const prefill = row.prompt_tokens * (lab.prefill_us_per_token / 1000);
  const job = { phase: "done", ttft: prefill + gap, itls: [] };
  fillItl(job, gap, row.output_tokens);
  return packLabs(lab, [job]);
}

function simulateCold(lab, repair) {
  const aMs = (lab.prefill_us_per_token / 1000) * (repair === "prefill" ? 0.5 : 1);
  const gap = lab.decode_us_per_token / 1000;
  const replicas = repair === "scale" ? 2 : 1;
  const load = repair === "warm" ? 0 : lab.cold_ms;
  const cap = repair === "admission" ? Math.ceil(lab.arrivals.length / 2) : lab.arrivals.length;
  const lanes = Array.from({ length: replicas }, () => load);
  const jobs = [];
  lab.arrivals.forEach((row, index) => {
    if (jobs.filter((job) => job.phase === "done").length >= cap && repair === "admission") {
      jobs.push({ phase: "rejected", ttft: null, itls: [] });
      return;
    }
    const lane = index % replicas;
    const start = Math.max(lanes[lane], row.arrival_time_ms);
    const doneAt = start + row.prompt_tokens * aMs;
    lanes[lane] = doneAt;
    const job = { phase: "done", ttft: doneAt + gap - row.arrival_time_ms, itls: [] };
    fillItl(job, gap, row.output_tokens);
    jobs.push(job);
  });
  return packLabs(lab, jobs);
}

function row(id, arrival, prompt, output, extra) {
  return { request_id: id, arrival_time_ms: arrival, prompt_tokens: prompt, output_tokens: output, ...extra };
}

function base(extra) {
  return {
    prefill_us_per_token: 20,
    decode_us_per_token: 2000,
    kv_slots: 8,
    slo: { ttft_ms: 500, itl_ms: 20 },
    ...extra,
  };
}

const queueRows = [row("q1", 0, 500, 16), row("q2", 10, 500, 16), row("q3", 20, 800, 32)];
const chunkRows = [row("long", 0, 4096, 16), row("s1", 5, 100, 8), row("s2", 6, 120, 8)];
const retryRows = [row("u1", 0, 600, 24), row("u2", 10, 400, 16)];
const many = Array.from({ length: 8 }, (_, i) => row(`g${i}`, i * 10, 400, 24));
const prefixRows = Array.from({ length: 4 }, (_, i) => row(`p${i}`, i * 20, 4200, 16));
const priorityRows = [
  row("job", 0, 1000, 2000, { priority: "batch" }),
  row("c1", 50, 300, 32, { priority: "interactive" }),
  row("c2", 60, 300, 32, { priority: "interactive" }),
];
const stickyRows = [
  row("s1", 0, 3200, 16, { user_id: 0 }),
  row("s2", 10, 3200, 16, { user_id: 1 }),
  row("s3", 20, 3200, 16, { user_id: 0 }),
];
const hogRows = [row("hog", 0, 500, 4000), row("c1", 20, 300, 32), row("c2", 30, 300, 32)];
const specRows = [row("s1", 0, 800, 64)];
const evictRows = Array.from({ length: 6 }, (_, i) => row(`e${i}`, i * 10, 2200, 16));
const loraRows = [0, 1, 0, 2, 1, 2].map((adapter, i) => row(`l${i}`, i * 10, 400, 16, { adapter }));
const scaleRows = [0, 10, 20, 600, 610].map((t, i) => row(`x${i}`, t, 500, 24));
const batchRows = [0, 40, 80, 120, 400].map((t, i) => row(`b${i}`, t, 400, 16));
const sharedRows = [0, 10, 20].map((t, i) => row(`p${i}`, t, 20000, 32, { shared_prefix_tokens: 16000 }));
const coldRows = Array.from({ length: 4 }, (_, i) => row(`c${i}`, 0, 500, 16));

const cases = [
  ["admit-a", simulateAdmit, base({ kind: "admit", kv_slots: 4, arrivals: many }), "a"],
  ["admit-c", simulateAdmit, base({ kind: "admit", kv_slots: 4, arrivals: many }), "c"],
  ["queue-base", simulateQueue, base({ kind: "queue", arrivals: queueRows }), null],
  ["queue-engines", simulateQueue, base({ kind: "queue", arrivals: queueRows }), "add_engines"],
  ["queue-hide", simulateQueue, base({ kind: "queue", arrivals: queueRows, effects: { a: "faster_prefill" } }), "a"],
  ["chunk-base", simulateChunk, base({ kind: "chunk", chunk_tokens: 2048, arrivals: chunkRows }), null],
  ["chunk-on", simulateChunk, base({ kind: "chunk", chunk_tokens: 2048, arrivals: chunkRows }), "chunk"],
  ["chunk-drop", simulateChunk, base({ kind: "chunk", arrivals: chunkRows }), "reject_shorts"],
  ["retry-base", simulateRetry, base({ kind: "retry", kv_slots: 4, retry_mul: 3, timeout_ms: 200, arrivals: retryRows }), null],
  ["retry-stop", simulateRetry, base({ kind: "retry", kv_slots: 4, retry_mul: 3, timeout_ms: 200, arrivals: retryRows }), "stop_retries"],
  ["retry-cap", simulateRetry, base({ kind: "retry_storm", kv_slots: 4, arrivals: retryRows }), "admit_cap"],
  ["goodput-base", simulateGoodput, base({ kind: "goodput", kv_slots: 4, arrivals: many }), null],
  ["goodput-cap", simulateGoodput, base({ kind: "goodput", kv_slots: 4, arrivals: many }), "cap4"],
  ["prefix-base", simulatePrefix, base({ kind: "prefix", prefix_tokens: 4000, arrivals: prefixRows }), null],
  ["prefix-on", simulatePrefix, base({ kind: "prefix", prefix_tokens: 4000, arrivals: prefixRows }), "enable_prefix"],
  ["priority-base", simulatePriority, base({ kind: "priority", arrivals: priorityRows }), null],
  ["priority-preempt", simulatePriority, base({ kind: "priority", arrivals: priorityRows }), "preempt_batch"],
  ["sticky-base", simulateSticky, base({ kind: "sticky", replicas: 3, prefix_tokens: 3000, arrivals: stickyRows }), null],
  ["sticky-pin", simulateSticky, base({ kind: "sticky", replicas: 3, prefix_tokens: 3000, arrivals: stickyRows }), "pin_user"],
  ["sticky-route", simulateSticky, base({ kind: "sticky", prefix_tokens: 3000, arrivals: stickyRows }), "sticky_route"],
  ["hog-base", simulateHog, base({ kind: "hog", max_output: 128, arrivals: hogRows }), null],
  ["hog-cap", simulateHog, base({ kind: "hog", max_output: 128, arrivals: hogRows }), "cap_output"],
  ["hog-preempt", simulateHog, base({ kind: "hog", arrivals: hogRows }), "preempt_long"],
  ["frag-base", simulateFrag, base({ kind: "frag", arrivals: many }), null],
  ["frag-paged", simulateFrag, base({ kind: "frag", arrivals: many }), "paged_blocks"],
  ["frag-buy", simulateFrag, base({ kind: "frag", arrivals: many }), "buy_memory"],
  ["frag-low", simulateFrag, base({ kind: "frag", arrivals: many }), "lower_batch"],
  ["spec-base", simulateSpec, base({ kind: "spec", draft_k: 4, accept: 0.2, arrivals: specRows }), null],
  ["spec-accept", simulateSpec, base({ kind: "spec", draft_k: 4, accept: 0.2, arrivals: specRows }), "raise_accept"],
  ["spec-off", simulateSpec, base({ kind: "spec", draft_k: 4, accept: 0.2, arrivals: specRows }), "disable_spec"],
  ["evict-base", simulateEvict, base({ kind: "evict", cache_blocks: 4, prefix_tokens: 2000, arrivals: evictRows }), null],
  ["evict-grow", simulateEvict, base({ kind: "evict", cache_blocks: 4, prefix_tokens: 2000, arrivals: evictRows }), "grow_cache"],
  ["lora-base", simulateLora, base({ kind: "lora", swap_ms: 40, unique_adapters: 4, arrivals: loraRows }), null],
  ["lora-queue", simulateLora, base({ kind: "lora", swap_ms: 40, arrivals: loraRows }), "queue_by_adapter"],
  ["lora-pin", simulateLora, base({ kind: "lora", swap_ms: 40, unique_adapters: 2, arrivals: loraRows }), "pin_adapters"],
  ["scale-base", simulateScale, base({ kind: "scale", kv_slots: 4, warm_ms: 3000, arrivals: scaleRows }), null],
  ["scale-prewarm", simulateScale, base({ kind: "scale", kv_slots: 4, warm_ms: 3000, arrivals: scaleRows }), "prewarm"],
  ["scale-fast", simulateScale, base({ kind: "scale", warm_ms: 3000, arrivals: scaleRows }), "faster_scale"],
  ["scale-shed", simulateScale, base({ kind: "scale", kv_slots: 2, arrivals: scaleRows }), "admit_shed"],
  ["batch-base", simulateBatch, base({ kind: "batch", arrivals: batchRows }), null],
  ["batch-cont", simulateBatch, base({ kind: "batch", arrivals: batchRows }), "continuous"],
  ["batch-big", simulateBatch, base({ kind: "batch", arrivals: batchRows }), "bigger_static"],
  ["labs-prefix", simulateLabsPrefix, base({ kind: "prefix", arrivals: sharedRows }), null],
  ["labs-prefix-hit", simulateLabsPrefix, base({ kind: "prefix", arrivals: sharedRows }), "prefix"],
  ["labs-spec", simulateLabsSpec, base({ kind: "spec", verify_us: 4000, draft_len: 8, accept_len: 1, draft_us_per_token: 500, arrivals: specRows }), null],
  ["labs-spec-off", simulateLabsSpec, base({ kind: "spec", verify_us: 4000, draft_len: 8, accept_len: 1, draft_us_per_token: 500, arrivals: specRows }), "off"],
  ["device", simulateDevice, base({ kind: "device", compute_ms: 2, link_ms: 20, local_ms: 4, arrivals: specRows }), null],
  ["device-fix", simulateDevice, base({ kind: "device", compute_ms: 2, link_ms: 20, local_ms: 4, arrivals: specRows }), "device"],
  ["cold", simulateCold, base({ kind: "cold", cold_ms: 3000, arrivals: coldRows }), null],
  ["cold-warm", simulateCold, base({ kind: "cold", cold_ms: 3000, arrivals: coldRows }), "warm"],
];

for (const [name, fn, scenario, repair] of cases) {
  const result = fn(scenario, repair);
  process.stdout.write(`${JSON.stringify({ name, repair, scenario, result })}\n`);
}
