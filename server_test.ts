// E2E tests for the distributed training coordination server (server.ts).
// Run: deno test --allow-net --allow-read --allow-run server_test.ts

import {
  assert,
  assertEquals,
  assertExists,
} from "https://deno.land/std@0.224.0/assert/mod.ts";

const BASE = "http://localhost:4000";
const WS_URL = "ws://localhost:4000/ws";

type UpdatePayload = {
  version: number;
  weights: number[];
  deadline_ms: number;
};

type UpdateMsg = {
  type: "update";
  train_id: string;
  payload: UpdatePayload;
};

// ─── Server process management ───────────────────────────────────────────────

let serverProcess: Deno.ChildProcess;

async function waitForServer(timeoutMs = 5000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${BASE}/train`);
      if (res.ok) return;
    } catch {
      // not ready yet
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error("Server did not start within timeout");
}

// ─── Helper utilities ────────────────────────────────────────────────────────

/** Returns a Promise that resolves with the next message of the given type. */
function waitForMsg(ws: WebSocket, type: string, timeoutMs = 5000): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      ws.removeEventListener("message", handler);
      reject(new Error(`Timeout (${timeoutMs}ms) waiting for message type: "${type}"`));
    }, timeoutMs);

    function handler(event: MessageEvent) {
      const msg = JSON.parse(event.data);
      if (msg.type === type) {
        clearTimeout(timer);
        ws.removeEventListener("message", handler);
        resolve(msg);
      }
    }
    ws.addEventListener("message", handler);
  });
}

/** Opens a WebSocket and returns it after receiving the first "update" message. */
async function joinClient(
  train_id: string,
): Promise<{ ws: WebSocket; update: UpdateMsg }> {
  const ws = new WebSocket(WS_URL);
  await new Promise<void>((resolve, reject) => {
    ws.addEventListener("open", () => resolve(), { once: true });
    ws.addEventListener("error", () => reject(new Error("WS connection failed")), { once: true });
  });
  const updatePromise = waitForMsg(ws, "update") as Promise<UpdateMsg>;
  ws.send(JSON.stringify({ type: "join", train_id }));
  const update = await updatePromise;
  return { ws, update };
}

/**
 * Simulates one training round with deterministic values.
 * delta[i] = lr * (clientIndex+1) * (i+1) * 0.01
 * tokens   = 100 + clientIndex * 10
 * loss     = 2.5 - clientIndex * 0.1
 */
function mockTrain(
  weights: number[],
  clientIndex: number,
  lr = 0.1,
): { delta: number[]; tokens: number; loss: number } {
  const delta = weights.map((_, i) => lr * (clientIndex + 1) * (i + 1) * 0.01);
  const tokens = 100 + clientIndex * 10;
  const loss = 2.5 - clientIndex * 0.1;
  return { delta, tokens, loss };
}

// ─── Setup / teardown ────────────────────────────────────────────────────────

Deno.test({ name: "setup: start server", sanitizeOps: false, sanitizeResources: false }, async () => {
  serverProcess = new Deno.Command(Deno.execPath(), {
    args: ["run", "--allow-net", "server.ts"],
    stdout: "null",
    stderr: "null",
  }).spawn();
  await waitForServer();
});

// ─── Test 1: Session creation ─────────────────────────────────────────────────

Deno.test("session creation", async () => {
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [0, 0, 0, 0], quorum: 4, round_duration_ms: 10000 }),
  });
  assertEquals(res.status, 201);
  const body = await res.json();
  assertExists(body.train_id);
  assertEquals(body.version, 0);
  const train_id: string = body.train_id;

  // GET /train lists the session
  const listRes = await fetch(`${BASE}/train`);
  const list = await listRes.json() as { train_id: string }[];
  assert(list.some((s) => s.train_id === train_id), "session not in list");

  // GET /train/:id returns full status
  const statusRes = await fetch(`${BASE}/train/${train_id}`);
  assertEquals(statusRes.status, 200);
  const status = await statusRes.json();
  assertEquals(status.train_id, train_id);
  assertEquals(status.version, 0);
  assertEquals(status.weights, [0, 0, 0, 0]);
  assertEquals(status.quorum, 4);
});

// ─── Test 2: Staggered joins receive current state ────────────────────────────

Deno.test({ name: "staggered joins receive current state", sanitizeOps: false, sanitizeResources: false }, async () => {
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [0, 0, 0, 0], quorum: 4, round_duration_ms: 60000 }),
  });
  const { train_id } = await res.json();

  const clients: { ws: WebSocket; update: UpdateMsg }[] = [];
  for (let i = 0; i < 4; i++) {
    if (i > 0) await new Promise((r) => setTimeout(r, 100));
    clients.push(await joinClient(train_id));
  }

  // All 4 clients must receive version 0 with the initial weights
  for (const { update } of clients) {
    assertEquals(update.payload.version, 0);
    assertEquals(update.payload.weights, [0, 0, 0, 0]);
  }

  for (const { ws } of clients) ws.close();
});

// ─── Test 3: Quorum-triggered round + weight math ─────────────────────────────

Deno.test({ name: "quorum-triggered round + weight math", sanitizeOps: false, sanitizeResources: false }, async () => {
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [0, 0, 0, 0], quorum: 4, round_duration_ms: 60000 }),
  });
  const { train_id } = await res.json();

  const clients = await Promise.all(
    Array.from({ length: 4 }, () => joinClient(train_id)),
  );

  // Register update listeners BEFORE publishing (version-1 update is the target)
  const updatePromises = clients.map(({ ws }) =>
    waitForMsg(ws, "update") as Promise<UpdateMsg>
  );

  // Each client publishes its deltas; collect ack promises
  const ackPromises = clients.map(({ ws }, i) => {
    const { delta, tokens, loss } = mockTrain([0, 0, 0, 0], i);
    ws.send(
      JSON.stringify({
        type: "publish",
        train_id,
        payload: {
          publish_id: `quorum-round0-c${i}`,
          base_version: 0,
          delta,
          tokens,
          loss,
          steps: 10,
        },
      }),
    );
    return waitForMsg(ws, "publish_ack") as Promise<{ submissions: number; quorum: number }>;
  });

  // Wait for all acks and all version-1 broadcasts
  const [acks, updates] = await Promise.all([
    Promise.all(ackPromises),
    Promise.all(updatePromises),
  ]);

  // Each ack must report a valid submission count
  for (const ack of acks) {
    assert(ack.submissions >= 1 && ack.submissions <= 4);
    assertEquals(ack.quorum, 4);
  }

  // All 4 clients must receive version 1
  for (const upd of updates) {
    assertEquals(upd.payload.version, 1);
  }

  // ── Weight math verification ──────────────────────────────────────────────
  // tokens_c = [100, 110, 120, 130]  total = 460
  // delta_c[i] = 0.1 * (c+1) * (i+1) * 0.01
  // w_new[i] = sum_c(tokens_c * delta_c[i]) / 460
  const totalTokens = 460;
  const expected = [0, 1, 2, 3].map((i) => {
    let acc = 0;
    for (let c = 0; c < 4; c++) {
      acc += (100 + c * 10) * (0.1 * (c + 1) * (i + 1) * 0.01);
    }
    return acc / totalTokens;
  });
  // expected ≈ [0.002609, 0.005217, 0.007826, 0.010435]

  const actualWeights = updates[0].payload.weights;
  assertEquals(actualWeights.length, 4);
  for (let i = 0; i < 4; i++) {
    assert(
      Math.abs(actualWeights[i] - expected[i]) < 1e-10,
      `w[${i}]: expected ${expected[i]}, got ${actualWeights[i]}`,
    );
  }

  for (const { ws } of clients) ws.close();
});

// ─── Test 4: Loss stored in loss_history ──────────────────────────────────────

Deno.test({ name: "loss stored in loss_history", sanitizeOps: false, sanitizeResources: false }, async () => {
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [0, 0, 0, 0], quorum: 4, round_duration_ms: 60000 }),
  });
  const { train_id } = await res.json();

  const clients = await Promise.all(
    Array.from({ length: 4 }, () => joinClient(train_id)),
  );
  const updatePromises = clients.map(({ ws }) => waitForMsg(ws, "update"));

  for (let i = 0; i < 4; i++) {
    const { delta, tokens, loss } = mockTrain([0, 0, 0, 0], i);
    clients[i].ws.send(
      JSON.stringify({
        type: "publish",
        train_id,
        payload: {
          publish_id: `loss-c${i}`,
          base_version: 0,
          delta,
          tokens,
          loss,
          steps: 10,
        },
      }),
    );
  }

  await Promise.all(updatePromises);

  const statusRes = await fetch(`${BASE}/train/${train_id}`);
  const status = await statusRes.json();

  assertEquals(status.loss_history.length, 1);
  assertEquals(status.loss_history[0].version, 0);

  // losses = [2.5, 2.4, 2.3, 2.2] → avg = 2.35
  const expectedAvgLoss = (2.5 + 2.4 + 2.3 + 2.2) / 4;
  assert(
    Math.abs(status.loss_history[0].avg_loss - expectedAvgLoss) < 1e-10,
    `avg_loss: expected ${expectedAvgLoss}, got ${status.loss_history[0].avg_loss}`,
  );

  for (const { ws } of clients) ws.close();
});

// ─── Test 5: Late join sees updated state ─────────────────────────────────────

Deno.test({ name: "late join sees updated state", sanitizeOps: false, sanitizeResources: false }, async () => {
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [0, 0, 0, 0], quorum: 4, round_duration_ms: 60000 }),
  });
  const { train_id } = await res.json();

  const clients = await Promise.all(
    Array.from({ length: 4 }, () => joinClient(train_id)),
  );
  const roundDonePromises = clients.map(({ ws }) => waitForMsg(ws, "update"));

  for (let i = 0; i < 4; i++) {
    const { delta, tokens, loss } = mockTrain([0, 0, 0, 0], i);
    clients[i].ws.send(
      JSON.stringify({
        type: "publish",
        train_id,
        payload: {
          publish_id: `late-join-c${i}`,
          base_version: 0,
          delta,
          tokens,
          loss,
          steps: 10,
        },
      }),
    );
  }

  // Wait for round 1 to complete on all 4 clients
  await Promise.all(roundDonePromises);

  // 5th client joins after the broadcast
  const { ws: lateWs, update: lateUpdate } = await joinClient(train_id);
  assertEquals(lateUpdate.payload.version, 1, "late joiner should see version 1");

  for (const { ws } of clients) ws.close();
  lateWs.close();
});

// ─── Test 6: Deadline-triggered round (timer path) ────────────────────────────

Deno.test({
  name: "deadline-triggered round",
  sanitizeOps: false,
  sanitizeResources: false,
}, async () => {
  // quorum: 5 but only 3 clients submit — round fires via deadline timer
  // grace_period_ms: 1000 so submit_request arrives at ~t=1500, round at ~t=2500
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [0, 0, 0, 0], quorum: 5, round_duration_ms: 2500, grace_period_ms: 1000 }),
  });
  const { train_id } = await res.json();

  const clients = await Promise.all(
    Array.from({ length: 3 }, () => joinClient(train_id)),
  );

  // Register submit_request and update listeners BEFORE publishing
  const submitRequestPromises = clients.map(({ ws }) =>
    waitForMsg(ws, "submit_request", 6000) as Promise<{ type: string; train_id: string; deadline_ms: number }>
  );
  const updatePromises = clients.map(({ ws }) =>
    waitForMsg(ws, "update", 8000) as Promise<UpdateMsg>
  );

  for (let i = 0; i < 3; i++) {
    const { delta, tokens, loss } = mockTrain([0, 0, 0, 0], i);
    clients[i].ws.send(
      JSON.stringify({
        type: "publish",
        train_id,
        payload: {
          publish_id: `deadline-c${i}`,
          base_version: 0,
          delta,
          tokens,
          loss,
          steps: 10,
        },
      }),
    );
  }

  // All 3 clients must receive submit_request before the deadline fires
  const submitRequests = await Promise.all(submitRequestPromises);
  for (const sr of submitRequests) {
    assertEquals(sr.type, "submit_request");
    assertEquals(sr.train_id, train_id);
    assert(sr.deadline_ms > Date.now(), "deadline_ms should be in the future when received");
  }

  // Then all 3 receive update when deadline fires
  const updates = await Promise.all(updatePromises);
  for (const upd of updates) {
    assertEquals(upd.payload.version, 1, "deadline path should advance to version 1");
  }

  for (const { ws } of clients) ws.close();
});

// ─── Test 7: Rejection cases ──────────────────────────────────────────────────

Deno.test({ name: "rejection cases", sanitizeOps: false, sanitizeResources: false }, async () => {
  // Create session and complete round 0 so version advances to 1
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [0, 0, 0, 0], quorum: 4, round_duration_ms: 60000 }),
  });
  const { train_id } = await res.json();

  const clients = await Promise.all(
    Array.from({ length: 4 }, () => joinClient(train_id)),
  );
  const roundDonePromises = clients.map(({ ws }) => waitForMsg(ws, "update"));

  for (let i = 0; i < 4; i++) {
    const { delta, tokens, loss } = mockTrain([0, 0, 0, 0], i);
    clients[i].ws.send(
      JSON.stringify({
        type: "publish",
        train_id,
        payload: {
          publish_id: `reject-setup-c${i}`,
          base_version: 0,
          delta,
          tokens,
          loss,
          steps: 10,
        },
      }),
    );
  }
  await Promise.all(roundDonePromises);
  // session.version is now 1

  // ── Case 1: Stale base_version (0 when server is at 1) ───────────────────
  const staleRejectPromise = waitForMsg(clients[0].ws, "publish_reject");
  clients[0].ws.send(
    JSON.stringify({
      type: "publish",
      train_id,
      payload: {
        publish_id: "stale-attempt",
        base_version: 0, // stale
        delta: [0.1, 0.1, 0.1, 0.1],
        tokens: 100,
        loss: 1.0,
        steps: 10,
      },
    }),
  );
  const staleReject = await staleRejectPromise as { type: string; reason: string };
  assertEquals(staleReject.reason, "version mismatch");

  // ── Case 2: Duplicate publish_id ─────────────────────────────────────────
  const dupId = "dup-publish-id-unique";
  // First submission with base_version: 1 should succeed
  const firstAckPromise = waitForMsg(clients[0].ws, "publish_ack");
  clients[0].ws.send(
    JSON.stringify({
      type: "publish",
      train_id,
      payload: {
        publish_id: dupId,
        base_version: 1,
        delta: [0.01, 0.01, 0.01, 0.01],
        tokens: 100,
        loss: 1.0,
        steps: 10,
      },
    }),
  );
  await firstAckPromise;

  // Second submission with the same publish_id → reject
  const dupRejectPromise = waitForMsg(clients[0].ws, "publish_reject");
  clients[0].ws.send(
    JSON.stringify({
      type: "publish",
      train_id,
      payload: {
        publish_id: dupId, // duplicate
        base_version: 1,
        delta: [0.01, 0.01, 0.01, 0.01],
        tokens: 100,
        loss: 1.0,
        steps: 10,
      },
    }),
  );
  const dupReject = await dupRejectPromise as { reason: string };
  assertEquals(dupReject.reason, "duplicate publish_id");

  // ── Case 3: Wrong delta length ────────────────────────────────────────────
  const deltaRejectPromise = waitForMsg(clients[1].ws, "publish_reject");
  clients[1].ws.send(
    JSON.stringify({
      type: "publish",
      train_id,
      payload: {
        publish_id: "wrong-delta-len",
        base_version: 1,
        delta: [0.01, 0.01], // only 2 elements; session has 4 weights
        tokens: 100,
        loss: 1.0,
        steps: 10,
      },
    }),
  );
  const deltaReject = await deltaRejectPromise as { reason: string };
  assertEquals(deltaReject.reason, "delta length mismatch");

  for (const { ws } of clients) ws.close();
});

// ─── Test 8: submit_request timing and late-joiner during grace period ────────

Deno.test({
  name: "submit_request: timing, payload, and late join during grace period",
  sanitizeOps: false,
  sanitizeResources: false,
}, async () => {
  // round_duration_ms: 2500, grace_period_ms: 1000
  // submit_request arrives at ~t=1500, round aggregates at ~t=2500
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [0, 0, 0, 0], quorum: 10, round_duration_ms: 2500, grace_period_ms: 1000 }),
  });
  const { train_id } = await res.json();

  // Verify GET /train/:id exposes grace_period_ms
  const status = await (await fetch(`${BASE}/train/${train_id}`)).json();
  assertEquals(status.grace_period_ms, 1000);
  assertEquals(status.submit_request_sent, false);

  const clients = await Promise.all(
    Array.from({ length: 2 }, () => joinClient(train_id)),
  );

  // Register submit_request listeners on both early clients
  const srPromises = clients.map(({ ws }) =>
    waitForMsg(ws, "submit_request", 6000) as Promise<{ deadline_ms: number }>
  );

  // Wait for grace period to begin, then a 3rd client joins mid-grace-period
  // It should immediately receive submit_request on join (server sends it in handleJoin)
  const [sr0] = await Promise.all([srPromises[0]]);  // wait for first to confirm grace started

  const { ws: lateWs, update: lateUpdate } = await joinClient(train_id);
  assertEquals(lateUpdate.payload.version, 0);

  // Late joiner should get submit_request immediately since grace period is active
  const lateSr = await waitForMsg(lateWs, "submit_request", 3000) as { deadline_ms: number; train_id: string };
  assertEquals(lateSr.train_id, train_id);
  assert(lateSr.deadline_ms > Date.now(), "late joiner's submit_request deadline should be in the future");
  // deadline_ms matches what early clients saw
  assertEquals(lateSr.deadline_ms, sr0.deadline_ms);

  // Also confirm GET /train/:id now shows submit_request_sent: true
  const statusMid = await (await fetch(`${BASE}/train/${train_id}`)).json();
  assertEquals(statusMid.submit_request_sent, true);

  // Have all clients (including late joiner) submit so the round fires at deadline
  const updatePromises = [
    ...clients.map(({ ws }) => waitForMsg(ws, "update", 8000)),
    waitForMsg(lateWs, "update", 8000),
  ];
  for (const [i, { ws }] of [...clients.entries()]) {
    const { delta, tokens, loss } = mockTrain([0, 0, 0, 0], i);
    ws.send(JSON.stringify({
      type: "publish",
      train_id,
      payload: { publish_id: `sr-test-c${i}`, base_version: 0, delta, tokens, loss, steps: 10 },
    }));
  }
  // late joiner also submits
  const { delta, tokens, loss } = mockTrain([0, 0, 0, 0], 2);
  lateWs.send(JSON.stringify({
    type: "publish",
    train_id,
    payload: { publish_id: "sr-test-late", base_version: 0, delta, tokens, loss, steps: 5 },
  }));

  const updates = await Promise.all(updatePromises);
  for (const upd of updates) {
    assertEquals((upd as UpdateMsg).payload.version, 1);
  }

  // After round completes, submit_request_sent should be reset
  const statusAfter = await (await fetch(`${BASE}/train/${train_id}`)).json();
  assertEquals(statusAfter.submit_request_sent, false);

  for (const { ws } of clients) ws.close();
  lateWs.close();
});

// ─── Test 9: Multiple rounds — weights accumulate correctly ──────────────────

Deno.test({ name: "multiple rounds accumulate weights correctly", sanitizeOps: false, sanitizeResources: false }, async () => {
  // quorum: 2, two clients run 3 full rounds
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [0, 0, 0, 0], quorum: 2, round_duration_ms: 60000 }),
  });
  const { train_id } = await res.json();

  const clients = await Promise.all(
    Array.from({ length: 2 }, () => joinClient(train_id)),
  );

  // Track the running weights so we can verify each round's math
  let currentWeights = [0, 0, 0, 0];

  for (let round = 0; round < 3; round++) {
    const baseVersion = round;

    // Register update listeners before publishing
    const updatePromises = clients.map(({ ws }) =>
      waitForMsg(ws, "update") as Promise<UpdateMsg>
    );

    // Both clients submit using the weights they received
    for (let c = 0; c < 2; c++) {
      const { delta, tokens, loss } = mockTrain(currentWeights, c);
      clients[c].ws.send(
        JSON.stringify({
          type: "publish",
          train_id,
          payload: {
            publish_id: `multi-round${round}-c${c}`,
            base_version: baseVersion,
            delta,
            tokens,
            loss,
            steps: 10,
          },
        }),
      );
    }

    const updates = await Promise.all(updatePromises);

    // Verify version bumped
    for (const upd of updates) {
      assertEquals(upd.payload.version, round + 1);
    }

    // Compute expected weights for this round
    // clients c0, c1 → tokens [100, 110], total = 210
    const total = 210;
    const expectedWeights = currentWeights.map((w, i) => {
      let acc = 0;
      for (let c = 0; c < 2; c++) {
        const { delta, tokens } = mockTrain(currentWeights, c);
        acc += tokens * delta[i];
      }
      return w + acc / total;
    });

    const actualWeights = updates[0].payload.weights;
    for (let i = 0; i < 4; i++) {
      assert(
        Math.abs(actualWeights[i] - expectedWeights[i]) < 1e-10,
        `round ${round + 1} w[${i}]: expected ${expectedWeights[i]}, got ${actualWeights[i]}`,
      );
    }

    currentWeights = [...actualWeights];
  }

  // loss_history should have 3 entries
  const statusRes = await fetch(`${BASE}/train/${train_id}`);
  const status = await statusRes.json();
  assertEquals(status.loss_history.length, 3);
  assertEquals(status.version, 3);

  for (const { ws } of clients) ws.close();
});

// ─── Test 9: Client discards stale weights and retries with latest ─────────────

Deno.test({ name: "client discards stale weights and retries with latest", sanitizeOps: false, sanitizeResources: false }, async () => {
  // Setup: quorum 2, client A joins early, round fires without A submitting.
  // A still holds v0 weights. A submits with base_version: 0 → rejected.
  // A reads the v1 update it received, retries with base_version: 1 → accepted.

  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [1, 2, 3, 4], quorum: 2, round_duration_ms: 60000 }),
  });
  const { train_id } = await res.json();

  // Client A: joins, gets v0 weights
  const { ws: wsA, update: updateA } = await joinClient(train_id);
  const staleWeights = updateA.payload.weights; // [1, 2, 3, 4]
  assertEquals(updateA.payload.version, 0);

  // Clients B and C trigger round 1 without A
  const { ws: wsB } = await joinClient(train_id);
  const { ws: wsC } = await joinClient(train_id);

  // A registers listener for the v1 broadcast before B/C submit
  const aUpdatePromise = waitForMsg(wsA, "update") as Promise<UpdateMsg>;
  const bUpdatePromise = waitForMsg(wsB, "update") as Promise<UpdateMsg>;

  for (const [i, ws] of [[0, wsB], [1, wsC]] as [number, WebSocket][]) {
    const { delta, tokens, loss } = mockTrain([1, 2, 3, 4], i);
    ws.send(
      JSON.stringify({
        type: "publish",
        train_id,
        payload: {
          publish_id: `discard-bc-c${i}`,
          base_version: 0,
          delta,
          tokens,
          loss,
          steps: 10,
        },
      }),
    );
  }

  // Wait for v1 broadcast — all three connected clients (A, B, C) receive it
  const [aV1Update] = await Promise.all([aUpdatePromise, bUpdatePromise]);
  assertEquals(aV1Update.payload.version, 1);
  const freshWeights = aV1Update.payload.weights; // weights after round 1

  // ── A tries to submit with stale base_version: 0 → rejected ───────────────
  const staleRejectPromise = waitForMsg(wsA, "publish_reject");
  wsA.send(
    JSON.stringify({
      type: "publish",
      train_id,
      payload: {
        publish_id: "discard-stale-a",
        base_version: 0, // stale
        delta: mockTrain(staleWeights, 2).delta,
        tokens: mockTrain(staleWeights, 2).tokens,
        loss: mockTrain(staleWeights, 2).loss,
        steps: 10,
      },
    }),
  );
  const staleReject = await staleRejectPromise as { reason: string };
  assertEquals(staleReject.reason, "version mismatch");

  // ── A discards stale weights, trains on freshWeights, submits with v1 → ack ─
  const { ws: wsD } = await joinClient(train_id); // need a 2nd client for quorum
  const freshAckPromise = waitForMsg(wsA, "publish_ack");
  const { delta, tokens, loss } = mockTrain(freshWeights, 2);
  wsA.send(
    JSON.stringify({
      type: "publish",
      train_id,
      payload: {
        publish_id: "discard-fresh-a",
        base_version: 1, // correct version
        delta,
        tokens,
        loss,
        steps: 10,
      },
    }),
  );
  const freshAck = await freshAckPromise as { submissions: number };
  assert(freshAck.submissions >= 1);

  wsA.close();
  wsB.close();
  wsC.close();
  wsD.close();
});

// ─── Test A: join_ack contains client_id ──────────────────────────────────────

Deno.test({ name: "join_ack contains client_id", sanitizeOps: false, sanitizeResources: false }, async () => {
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [0, 0], quorum: 2, round_duration_ms: 60000 }),
  });
  const { train_id } = await res.json();

  const ws = new WebSocket(WS_URL);
  await new Promise<void>((resolve, reject) => {
    ws.addEventListener("open", () => resolve(), { once: true });
    ws.addEventListener("error", () => reject(new Error("WS connection failed")), { once: true });
  });

  const ackPromise = waitForMsg(ws, "join_ack") as Promise<{ client_id: string; train_id: string }>;
  ws.send(JSON.stringify({ type: "join", train_id, name: "device-alpha" }));
  const ack = await ackPromise;

  assertEquals(ack.train_id, train_id);
  assertExists(ack.client_id);
  assert(typeof ack.client_id === "string" && ack.client_id.length > 0, "client_id must be a non-empty string");

  ws.close();
});

// ─── Test B: clients_changed on join and leave ────────────────────────────────

Deno.test({ name: "clients_changed on join and leave", sanitizeOps: false, sanitizeResources: false }, async () => {
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [0, 0], quorum: 10, round_duration_ms: 60000 }),
  });
  const { train_id } = await res.json();

  // Client 1 joins
  const ws1 = new WebSocket(WS_URL);
  await new Promise<void>((r) => ws1.addEventListener("open", () => r(), { once: true }));
  ws1.send(JSON.stringify({ type: "join", train_id, name: "c1" }));
  // Wait for initial update + join_ack + clients_changed
  await waitForMsg(ws1, "join_ack");
  const cc1 = await waitForMsg(ws1, "clients_changed") as { clients: { client_id: string; name: string }[] };
  assertEquals(cc1.clients.length, 1);
  assertEquals(cc1.clients[0].name, "c1");

  // Client 2 joins — client 1 should receive clients_changed with 2 entries
  const cc1After2Promise = waitForMsg(ws1, "clients_changed") as Promise<{ clients: { client_id: string; name: string }[] }>;
  const ws2 = new WebSocket(WS_URL);
  await new Promise<void>((r) => ws2.addEventListener("open", () => r(), { once: true }));
  ws2.send(JSON.stringify({ type: "join", train_id, name: "c2" }));
  await waitForMsg(ws2, "join_ack");

  const cc1After2 = await cc1After2Promise;
  assertEquals(cc1After2.clients.length, 2);

  // Client 3 joins
  const cc1After3Promise = waitForMsg(ws1, "clients_changed") as Promise<{ clients: { client_id: string }[] }>;
  const ws3 = new WebSocket(WS_URL);
  await new Promise<void>((r) => ws3.addEventListener("open", () => r(), { once: true }));
  ws3.send(JSON.stringify({ type: "join", train_id, name: "c3" }));
  await waitForMsg(ws3, "join_ack");

  const cc1After3 = await cc1After3Promise;
  assertEquals(cc1After3.clients.length, 3);

  // Client 2 leaves — clients 1 and 3 should each receive clients_changed with 2 entries
  const cc1AfterLeavePromise = waitForMsg(ws1, "clients_changed") as Promise<{ clients: { client_id: string }[] }>;
  const cc3AfterLeavePromise = waitForMsg(ws3, "clients_changed") as Promise<{ clients: { client_id: string }[] }>;
  ws2.close();

  const [cc1AfterLeave, cc3AfterLeave] = await Promise.all([cc1AfterLeavePromise, cc3AfterLeavePromise]);
  assertEquals(cc1AfterLeave.clients.length, 2);
  assertEquals(cc3AfterLeave.clients.length, 2);

  ws1.close();
  ws3.close();
});

// ─── Test C: update broadcast includes avg_loss after round ──────────────────

Deno.test({ name: "update broadcast includes avg_loss after round", sanitizeOps: false, sanitizeResources: false }, async () => {
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [0, 0, 0, 0], quorum: 2, round_duration_ms: 60000 }),
  });
  const { train_id } = await res.json();

  const clients = await Promise.all([joinClient(train_id), joinClient(train_id)]);
  const updatePromises = clients.map(({ ws }) =>
    waitForMsg(ws, "update") as Promise<UpdateMsg & { payload: { avg_loss: number } }>
  );

  // loss[0] = 2.5, loss[1] = 2.4 → avg = 2.45
  for (let i = 0; i < 2; i++) {
    const { delta, tokens, loss } = mockTrain([0, 0, 0, 0], i);
    clients[i].ws.send(JSON.stringify({
      type: "publish",
      train_id,
      payload: { publish_id: `avg-loss-c${i}`, base_version: 0, delta, tokens, loss, steps: 10 },
    }));
  }

  const updates = await Promise.all(updatePromises);
  const expectedAvgLoss = (2.5 + 2.4) / 2;
  for (const upd of updates) {
    assert(upd.payload.avg_loss != null, "avg_loss should be present in update payload");
    assert(
      Math.abs(upd.payload.avg_loss - expectedAvgLoss) < 1e-10,
      `avg_loss: expected ${expectedAvgLoss}, got ${upd.payload.avg_loss}`,
    );
  }

  for (const { ws } of clients) ws.close();
});

// ─── Test D: Full federated lifecycle (bootstrap + 2 rounds with grace period) ─
//
// Models the exact browser flow:
//   1. POST /train with weights: [] (empty — server bootstraps on first publish)
//   2. Two clients join, both receive version 0 update with empty weights
//   3. Grace period fires → both receive submit_request
//   4. Both publish deltas → server bootstraps + completes round 1
//   5. Both receive version 1 update with new aggregated weights
//   6. Both train again from new weights and publish → round 2 completes
//   7. Both receive version 2 update

Deno.test({
  name: "full federated lifecycle: bootstrap + 2 rounds via grace period",
  sanitizeOps: false,
  sanitizeResources: false,
}, async () => {
  // round_duration_ms: 3000, grace_period_ms: 1000
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [], quorum: 2, round_duration_ms: 3000, grace_period_ms: 1000 }),
  });
  assertEquals(res.status, 201);
  const { train_id } = await res.json();

  // Both clients join and receive initial state (empty weights, version 0)
  const [clientA, clientB] = await Promise.all([
    joinClient(train_id),
    joinClient(train_id),
  ]);
  assertEquals(clientA.update.payload.version, 0);
  assertEquals(clientB.update.payload.version, 0);
  // Server was created with empty weights — initial update returns whatever server has
  assert(Array.isArray(clientA.update.payload.weights));
  assert(Array.isArray(clientB.update.payload.weights));

  // ── Round 1: wait for grace period → submit → receive new weights ──────────
  // Register both listeners before awaiting — broadcast fires to both simultaneously
  const [srA, srB] = await Promise.all([
    waitForMsg(clientA.ws, "submit_request", 6000) as Promise<{ deadline_ms: number }>,
    waitForMsg(clientB.ws, "submit_request", 6000) as Promise<{ deadline_ms: number }>,
  ]);
  assert(srA.deadline_ms > Date.now(), `deadline_ms ${srA.deadline_ms} should be > now ${Date.now()}`);
  assertEquals(srA.deadline_ms, srB.deadline_ms, "both clients see same deadline");

  // Both publish deltas (bootstraps server weight dimensions to 4)
  const round1UpdateA = waitForMsg(clientA.ws, "update", 8000) as Promise<UpdateMsg>;
  const round1UpdateB = waitForMsg(clientB.ws, "update", 8000) as Promise<UpdateMsg>;

  const deltaLen = 4;
  const mockWeights = [0, 0, 0, 0];
  for (const [i, { ws }] of [clientA, clientB].entries()) {
    const { delta, tokens, loss } = mockTrain(mockWeights, i);
    ws.send(JSON.stringify({
      type: "publish",
      train_id,
      payload: { publish_id: `full-r1-c${i}`, base_version: 0, delta, tokens, loss, steps: 10 },
    }));
  }

  const [r1A, r1B] = await Promise.all([round1UpdateA, round1UpdateB]);
  assertEquals(r1A.payload.version, 1, "round 1 must advance to version 1");
  assertEquals(r1B.payload.version, 1);
  assertEquals(r1A.payload.weights.length, deltaLen, "bootstrapped weight length must match delta");
  // avg_loss must be present (non-null) after a completed round
  assert((r1A.payload as unknown as { avg_loss: number }).avg_loss != null, "avg_loss must be present after round 1");

  // ── Round 2: train from new weights → submit → receive version 2 ──────────
  const newWeights = r1A.payload.weights;

  const [sr2A, sr2B] = await Promise.all([
    waitForMsg(clientA.ws, "submit_request", 6000) as Promise<{ deadline_ms: number }>,
    waitForMsg(clientB.ws, "submit_request", 6000) as Promise<{ deadline_ms: number }>,
  ]);
  assertEquals(sr2A.deadline_ms, sr2B.deadline_ms, "round 2 deadlines match");

  const round2UpdateA = waitForMsg(clientA.ws, "update", 8000) as Promise<UpdateMsg>;
  const round2UpdateB = waitForMsg(clientB.ws, "update", 8000) as Promise<UpdateMsg>;

  for (const [i, { ws }] of [clientA, clientB].entries()) {
    const { delta, tokens, loss } = mockTrain(newWeights, i);
    ws.send(JSON.stringify({
      type: "publish",
      train_id,
      payload: { publish_id: `full-r2-c${i}`, base_version: 1, delta, tokens, loss, steps: 10 },
    }));
  }

  const [r2A, r2B] = await Promise.all([round2UpdateA, round2UpdateB]);
  assertEquals(r2A.payload.version, 2, "round 2 must advance to version 2");
  assertEquals(r2B.payload.version, 2);

  // Verify weight math for round 2: token-weighted avg of deltas applied to r1 weights
  const totalTokens = 210; // 100 + 110
  const expectedR2 = newWeights.map((w, i) => {
    let acc = 0;
    for (let c = 0; c < 2; c++) {
      const { delta, tokens } = mockTrain(newWeights, c);
      acc += tokens * delta[i];
    }
    return w + acc / totalTokens;
  });
  for (let i = 0; i < deltaLen; i++) {
    assert(
      Math.abs(r2A.payload.weights[i] - expectedR2[i]) < 1e-10,
      `round 2 w[${i}]: expected ${expectedR2[i]}, got ${r2A.payload.weights[i]}`,
    );
  }

  // GET /train/:id reflects version 2 and 2 loss_history entries
  const status = await fetch(`${BASE}/train/${train_id}`).then(r => r.json());
  assertEquals(status.version, 2);
  assertEquals(status.loss_history.length, 2);

  clientA.ws.close();
  clientB.ws.close();
});

// ─── Test E: DELETE session broadcasts session_closed ─────────────────────────

Deno.test({ name: "DELETE session broadcasts session_closed and removes session", sanitizeOps: false, sanitizeResources: false }, async () => {
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [1, 2, 3], quorum: 2, round_duration_ms: 60000 }),
  });
  const { train_id } = await res.json();

  // Two clients join
  const [c1, c2] = await Promise.all([joinClient(train_id), joinClient(train_id)]);

  // Register session_closed listeners before deleting
  const sc1 = waitForMsg(c1.ws, "session_closed", 3000) as Promise<{ train_id: string }>;
  const sc2 = waitForMsg(c2.ws, "session_closed", 3000) as Promise<{ train_id: string }>;

  // DELETE the session
  const delRes = await fetch(`${BASE}/train/${train_id}`, { method: "DELETE" });
  assertEquals(delRes.status, 200);
  const delBody = await delRes.json();
  assertEquals(delBody.deleted, train_id);

  // Both clients receive session_closed
  const [msg1, msg2] = await Promise.all([sc1, sc2]);
  assertEquals(msg1.train_id, train_id);
  assertEquals(msg2.train_id, train_id);

  // Session is gone from the list
  const listRes = await fetch(`${BASE}/train`);
  const list = await listRes.json() as { train_id: string }[];
  assert(!list.some((s) => s.train_id === train_id), "deleted session must not appear in list");

  // GET /train/:id returns 404
  const getRes = await fetch(`${BASE}/train/${train_id}`);
  assertEquals(getRes.status, 404);

  // DELETE again returns 404
  const del2Res = await fetch(`${BASE}/train/${train_id}`, { method: "DELETE" });
  assertEquals(del2Res.status, 404);

  c1.ws.close();
  c2.ws.close();
});

// ─── Test F: WS error handling — invalid messages ─────────────────────────────

Deno.test({ name: "WS error handling: missing train_id, unknown type, invalid session", sanitizeOps: false, sanitizeResources: false }, async () => {
  const ws = new WebSocket(WS_URL);
  await new Promise<void>((resolve, reject) => {
    ws.addEventListener("open", () => resolve(), { once: true });
    ws.addEventListener("error", () => reject(new Error("WS connection failed")), { once: true });
  });

  // ── Case 1: message missing train_id ─────────────────────────────────────
  const err1 = waitForMsg(ws, "error", 2000) as Promise<{ message: string }>;
  ws.send(JSON.stringify({ type: "join" })); // no train_id
  const e1 = await err1;
  assertEquals(e1.message, "missing train_id");

  // ── Case 2: join non-existent session ─────────────────────────────────────
  const err2 = waitForMsg(ws, "error", 2000) as Promise<{ message: string; train_id: string }>;
  ws.send(JSON.stringify({ type: "join", train_id: "00000000-0000-0000-0000-000000000000" }));
  const e2 = await err2;
  assertEquals(e2.message, "session not found");

  // ── Case 3: unknown message type ──────────────────────────────────────────
  // Need a valid session for train_id requirement
  const sessRes = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights: [0], quorum: 1, round_duration_ms: 60000 }),
  });
  const { train_id } = await sessRes.json();

  const err3 = waitForMsg(ws, "error", 2000) as Promise<{ message: string }>;
  ws.send(JSON.stringify({ type: "ping", train_id }));
  const e3 = await err3;
  assert(e3.message.includes("unknown type"), `expected 'unknown type' error, got: ${e3.message}`);

  // ── Case 4: invalid JSON ───────────────────────────────────────────────────
  const err4 = waitForMsg(ws, "error", 2000) as Promise<{ message: string }>;
  ws.send("not json at all");
  const e4 = await err4;
  assertEquals(e4.message, "invalid JSON");

  ws.close();
});

// ─── Teardown ─────────────────────────────────────────────────────────────────

Deno.test({ name: "teardown: stop server", sanitizeOps: false, sanitizeResources: false }, () => {
  serverProcess.kill();
});
