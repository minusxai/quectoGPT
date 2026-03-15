// Distributed Training Coordination Server
// Coordinates browser clients for federated averaging of GPT model weights.
// Run: deno run --allow-net server.ts  (listens on :4000)

type TrainSession = {
  train_id: string;
  version: number;
  weights: number[];
  quorum: number;
  round_duration_ms: number;
  grace_period_ms: number;
  deadline_ms: number;
  submit_request_sent: boolean;
  clients: Set<WebSocket>;
  client_list: Map<WebSocket, { id: string; name: string; joined_at: number }>;
  delta_accumulator: number[];
  token_accumulator: number;
  submissions: number;
  publish_ids_seen: Set<string>;
  loss_sum: number;
  loss_count: number;
  loss_history: { version: number; avg_loss: number }[];
};

type PublishPayload = {
  publish_id: string;
  base_version: number;
  delta: number[];
  steps: number;
  tokens: number;
  loss: number;
};

const sessions = new Map<string, TrainSession>();

function randomId(): string {
  return crypto.randomUUID();
}

function send(ws: WebSocket, data: unknown) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(data));
  }
}

function broadcastClientsChanged(session: TrainSession) {
  const clients = [...session.client_list.values()].map((c) => ({
    client_id: c.id,
    name: c.name,
    joined_at: c.joined_at,
  }));
  const msg = { type: "clients_changed", train_id: session.train_id, clients };
  for (const ws of session.clients) {
    if (ws.readyState === WebSocket.OPEN) send(ws, msg);
    else session.clients.delete(ws);
  }
}

function tryCompleteRound(session: TrainSession) {
  const now = Date.now();

  if (session.token_accumulator === 0) {
    session.deadline_ms = now + session.round_duration_ms;
    return;
  }

  // Compute token-weighted global delta and apply to weights
  for (let i = 0; i < session.weights.length; i++) {
    const globalDelta = session.delta_accumulator[i] / session.token_accumulator;
    session.weights[i] += globalDelta;
  }

  const avg_loss = session.loss_sum / session.loss_count;
  session.loss_history.push({ version: session.version, avg_loss });

  session.version += 1;
  session.delta_accumulator = new Array(session.weights.length).fill(0);
  session.token_accumulator = 0;
  session.submissions = 0;
  session.publish_ids_seen.clear();
  session.loss_sum = 0;
  session.loss_count = 0;
  session.deadline_ms = now + session.round_duration_ms;
  session.submit_request_sent = false;

  const msg = {
    type: "update",
    train_id: session.train_id,
    payload: {
      version: session.version,
      weights: session.weights,
      deadline_ms: session.deadline_ms,
      avg_loss,
    },
  };

  // Broadcast to all connected clients, pruning closed sockets
  for (const ws of session.clients) {
    if (ws.readyState === WebSocket.OPEN) {
      send(ws, msg);
    } else {
      session.clients.delete(ws);
    }
  }
}

function handleJoin(ws: WebSocket, train_id: string, name: string) {
  const session = sessions.get(train_id);
  if (!session) {
    send(ws, { type: "error", message: "session not found", train_id });
    return;
  }

  const clientId = randomId();
  session.clients.add(ws);
  session.client_list.set(ws, { id: clientId, name, joined_at: Date.now() });

  const lastAvgLoss = session.loss_history.length > 0
    ? session.loss_history.at(-1)!.avg_loss
    : null;

  send(ws, {
    type: "update",
    train_id,
    payload: {
      version: session.version,
      weights: session.weights,
      deadline_ms: session.deadline_ms,
      avg_loss: lastAvgLoss,
    },
  });

  send(ws, { type: "join_ack", train_id, client_id: clientId });

  // If already in grace period, notify immediately so the client can wrap up
  if (session.submit_request_sent) {
    send(ws, { type: "submit_request", train_id, deadline_ms: session.deadline_ms });
  }

  broadcastClientsChanged(session);
}

function handlePublish(ws: WebSocket, train_id: string, payload: PublishPayload) {
  const session = sessions.get(train_id);

  if (!session) {
    send(ws, { type: "publish_reject", reason: "session not found" });
    return;
  }
  if (payload.base_version !== session.version) {
    send(ws, { type: "publish_reject", reason: "version mismatch", expected: session.version, got: payload.base_version });
    return;
  }
  if (session.publish_ids_seen.has(payload.publish_id)) {
    send(ws, { type: "publish_reject", reason: "duplicate publish_id" });
    return;
  }
  if (payload.delta.length !== session.weights.length) {
    send(ws, { type: "publish_reject", reason: "delta length mismatch", expected: session.weights.length, got: payload.delta.length });
    return;
  }
  if (payload.tokens <= 0) {
    send(ws, { type: "publish_reject", reason: "tokens must be positive" });
    return;
  }

  for (let i = 0; i < session.weights.length; i++) {
    session.delta_accumulator[i] += payload.tokens * payload.delta[i];
  }
  session.token_accumulator += payload.tokens;
  session.loss_sum += payload.loss;
  session.loss_count += 1;
  session.submissions += 1;
  session.publish_ids_seen.add(payload.publish_id);

  send(ws, { type: "publish_ack", publish_id: payload.publish_id, submissions: session.submissions, quorum: session.quorum });

  if (session.submissions >= session.quorum) {
    tryCompleteRound(session);
  }
}

// Timer: check grace period and deadlines every 250ms for timely submit_request delivery
setInterval(() => {
  const now = Date.now();
  for (const session of sessions.values()) {
    // Enter grace period: broadcast submit_request so clients can wrap up their current step
    if (!session.submit_request_sent && now >= session.deadline_ms - session.grace_period_ms) {
      session.submit_request_sent = true;
      const msg = { type: "submit_request", train_id: session.train_id, deadline_ms: session.deadline_ms };
      for (const ws of session.clients) {
        if (ws.readyState === WebSocket.OPEN) send(ws, msg);
        else session.clients.delete(ws);
      }
    }

    // Deadline: aggregate whatever was submitted, or roll the deadline forward
    if (now >= session.deadline_ms) {
      if (session.submissions > 0) {
        tryCompleteRound(session);
      } else {
        session.deadline_ms = now + session.round_duration_ms;
        session.submit_request_sent = false;
      }
    }
  }
}, 250);

// WebSocket handler
function handleWebSocket(req: Request): Response {
  const { socket: ws, response } = Deno.upgradeWebSocket(req);

  ws.onmessage = (event) => {
    let msg: { type: string; train_id?: string; payload?: PublishPayload };
    try {
      msg = JSON.parse(event.data);
    } catch {
      send(ws, { type: "error", message: "invalid JSON" });
      return;
    }

    const { type, train_id } = msg;
    if (!train_id) {
      send(ws, { type: "error", message: "missing train_id" });
      return;
    }

    if (type === "join") {
      const name = (msg as { name?: string }).name?.trim().slice(0, 32) ?? '';
      handleJoin(ws, train_id, name);
    } else if (type === "publish") {
      if (!msg.payload) {
        send(ws, { type: "error", message: "missing payload" });
        return;
      }
      handlePublish(ws, train_id, msg.payload);
    } else {
      send(ws, { type: "error", message: `unknown type: ${type}` });
    }
  };

  ws.onclose = () => {
    for (const session of sessions.values()) {
      session.clients.delete(ws);
      session.client_list.delete(ws);
      if (session.clients.size > 0) broadcastClientsChanged(session);
    }
  };

  ws.onerror = () => {
    for (const session of sessions.values()) {
      session.clients.delete(ws);
      session.client_list.delete(ws);
      if (session.clients.size > 0) broadcastClientsChanged(session);
    }
  };

  return response;
}

// HTTP handler
Deno.serve({ port: 4000 }, async (req) => {
  const url = new URL(req.url);
  const { pathname } = url;
  const method = req.method;

  // WebSocket upgrade
  if (pathname === "/ws") {
    return handleWebSocket(req);
  }

  // POST /train — create session
  if (method === "POST" && pathname === "/train") {
    let body: { weights?: number[]; quorum?: number; round_duration_ms?: number; grace_period_ms?: number } = {};
    try {
      body = await req.json();
    } catch {
      // use defaults
    }

    const weights = body.weights ?? [0, 0, 0, 0];
    const quorum = body.quorum ?? 2;
    const round_duration_ms = body.round_duration_ms ?? 60000;
    // Default grace period: 20% of round duration, minimum 500ms
    const grace_period_ms = body.grace_period_ms ?? Math.max(500, Math.round(round_duration_ms * 0.2));
    const train_id = randomId();
    const now = Date.now();

    const session: TrainSession = {
      train_id,
      version: 0,
      weights: [...weights],
      quorum,
      round_duration_ms,
      grace_period_ms,
      deadline_ms: now + round_duration_ms,
      submit_request_sent: false,
      clients: new Set(),
      client_list: new Map(),
      delta_accumulator: new Array(weights.length).fill(0),
      token_accumulator: 0,
      submissions: 0,
      publish_ids_seen: new Set(),
      loss_sum: 0,
      loss_count: 0,
      loss_history: [],
    };

    sessions.set(train_id, session);
    return Response.json({ train_id, version: session.version }, { status: 201 });
  }

  // GET /train — list sessions
  if (method === "GET" && pathname === "/train") {
    const list = [...sessions.values()].map((s) => ({
      train_id: s.train_id,
      version: s.version,
      clients: s.clients.size,
      deadline_ms: s.deadline_ms,
      submissions: s.submissions,
      quorum: s.quorum,
    }));
    return Response.json(list);
  }

  // GET /train/:id — session status
  if (method === "GET" && pathname.startsWith("/train/")) {
    const id = pathname.slice("/train/".length);
    const session = sessions.get(id);
    if (!session) {
      return Response.json({ error: "not found" }, { status: 404 });
    }
    return Response.json({
      train_id: session.train_id,
      version: session.version,
      weights: session.weights,
      quorum: session.quorum,
      round_duration_ms: session.round_duration_ms,
      grace_period_ms: session.grace_period_ms,
      deadline_ms: session.deadline_ms,
      submit_request_sent: session.submit_request_sent,
      clients: session.clients.size,
      client_list: [...session.client_list.values()].map((c) => ({
        client_id: c.id,
        name: c.name,
        joined_at: c.joined_at,
      })),
      submissions: session.submissions,
      token_accumulator: session.token_accumulator,
      loss_history: session.loss_history,
    });
  }

  // DELETE /train/:id — remove session
  if (method === "DELETE" && pathname.startsWith("/train/")) {
    const id = pathname.slice("/train/".length);
    if (!sessions.has(id)) {
      return Response.json({ error: "not found" }, { status: 404 });
    }
    const session = sessions.get(id)!;
    for (const ws of session.clients) {
      send(ws, { type: "session_closed", train_id: id });
    }
    sessions.delete(id);
    return Response.json({ deleted: id });
  }

  return Response.json({ error: "not found" }, { status: 404 });
});
