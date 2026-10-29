"use strict";
const $ = id => document.getElementById(id);
const sides = ["先手", "後手"];
const ranks = "一二三四五六七八九";
let state = null, selection = null, busy = false, promotionOptions = [];

async function api(path, data) {
  const options = data === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({...data, version: state.version})};
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "接続に失敗しました。");
  return body;
}

function candidates() {
  if (!selection || !state) return [];
  return state.legal.filter(move => selection.drop ? move.startsWith(selection.drop + "*") : move.startsWith(selection.square));
}

function destination(move) { return move.slice(2, 4); }

function render() {
  if (!state) return;
  const human = state.human, opponent = 1 - human;
  $("human-label").textContent = `${sides[human]} · あなた`;
  $("opponent-label").textContent = `${sides[opponent]} · AI`;
  $("side").value = String(human);
  $("status").textContent = state.result ? state.result.text : state.turn !== human ? `${sides[state.turn]}・AIが考えています…` : busy ? "指し手を確認しています…" : `${sides[human]}・あなたの手番です`;
  $("detail").textContent = state.result ? "「もう一局」で新しい対局を始められます。" : state.check ? "王手です。玉を守る手を選んでください。" : "駒をタップすると、指せるマスが光ります。";
  $("new").disabled = busy;
  $("side").disabled = busy;
  $("resign").disabled = busy || !!state.result;
  for (const [id, color] of [["opponent-hand", opponent], ["human-hand", human]]) {
    const hand = $(id); hand.replaceChildren();
    if (!state.hands[color].length) {const empty = document.createElement("span"); empty.className = "empty"; empty.textContent = "なし"; hand.append(empty);}
    for (const piece of state.hands[color]) {
      const button = document.createElement("button");
      button.type = "button"; button.dataset.drop = piece.kind;
      button.disabled = color !== human || busy || !!state.result || state.turn !== human;
      button.setAttribute("aria-label", `${sides[color]}の持ち駒 ${piece.name} ${piece.count}枚`);
      button.classList.toggle("selected", color === human && selection?.drop === piece.kind);
      button.setAttribute("aria-pressed", String(color === human && selection?.drop === piece.kind));
      button.append(document.createTextNode(piece.name));
      const count = document.createElement("small"); count.textContent = `×${piece.count}`; button.append(count);
      button.onclick = () => {selection = selection?.drop === piece.kind ? null : {drop: piece.kind}; render();}; hand.append(button);
    }
  }
  $("files").replaceChildren(...Array.from({length: 9}, (_, i) => {const span = document.createElement("span"); span.textContent = human === 0 ? 9 - i : i + 1; return span;}));
  const board = $("board"); board.replaceChildren();
  const squares = human === 0 ? state.board : [...state.board].reverse();
  const legalDestinations = new Set(candidates().map(destination));
  const last = state.history.at(-1)?.usi;
  for (const cell of squares) {
    const button = document.createElement("button"); button.type = "button"; button.className = "square"; button.dataset.square = cell.square;
    const coordinate = `${cell.square[0]}${ranks[cell.square.charCodeAt(1) - 97]}`;
    button.setAttribute("aria-label", `${coordinate} ${cell.piece ? sides[cell.color] + "の" + cell.piece : "空きマス"}`);
    button.disabled = busy || !!state.result || state.turn !== human;
    button.classList.toggle("selected", selection?.square === cell.square);
    button.setAttribute("aria-pressed", String(selection?.square === cell.square));
    button.classList.toggle("legal", legalDestinations.has(cell.square));
    button.classList.toggle("last", !!last && destination(last) === cell.square);
    if (cell.piece) {const piece = document.createElement("span"); piece.className = "piece"; piece.classList.toggle("opponent", cell.color !== human); piece.classList.toggle("promoted", cell.promoted); piece.classList.toggle("long", cell.piece.length > 1); piece.textContent = cell.piece; button.append(piece);}
    button.onclick = () => chooseSquare(cell); board.append(button);
  }
  $("selection").textContent = selection ? selection.drop ? "持ち駒を打つマスを選んでください。" : "光ったマスを選んでください。同じ駒を押すと選択を解除します。" : "盤上の駒、または持ち駒を選んでください。";
  $("move-count").textContent = `${state.history.length}手`;
  $("empty-history").hidden = state.history.length > 0;
  const history = $("history"); history.replaceChildren();
  for (const move of state.history) {const li = document.createElement("li"); li.textContent = move.label; const code = document.createElement("code"); code.textContent = move.usi; li.append(code); history.append(li);}
  history.scrollTop = history.scrollHeight;
}

function chooseSquare(cell) {
  const options = candidates().filter(move => destination(move) === cell.square);
  if (options.length > 1) {promotionOptions = options; $("promotion").showModal(); return;}
  if (options.length === 1) {play(options[0]); return;}
  if (cell.color === state.human) {selection = selection?.square === cell.square ? null : {square: cell.square}; $("error").textContent = "";}
  else if (selection) {$("error").textContent = "そのマスには指せません。光ったマスを選んでください。";}
  render();
}

async function act(path, data) {
  if (busy || !state) return;
  busy = true; selection = null; $("error").textContent = ""; $("retry").hidden = true; render();
  try {
    state = await api(path, data); render();
    if (!state.result && state.turn !== state.human) {state = await api("/api/ai", {});}
  } catch (error) {$("error").textContent = `${error.message} 「再接続する」で現在の盤面を確認できます。`; $("retry").hidden = false;}
  finally {busy = false; render();}
}

function play(move) {return act("/api/move", {move});}
$("promote").onclick = () => {$("promotion").close(); play(promotionOptions.find(move => move.endsWith("+")));};
$("unpromote").onclick = () => {$("promotion").close(); play(promotionOptions.find(move => !move.endsWith("+")));};
$("cancel-promotion").onclick = () => $("promotion").close();
$("new").onclick = () => {const human = Number($("side").value); if (!state.history.length || state.result || confirm("いまの対局を終了して、もう一局始めますか？")) act("/api/new", {human});};
$("resign").onclick = () => {if (confirm("投了して対局を終了しますか？")) act("/api/resign", {});};

async function connect() {
  if (busy) return;
  busy = true; $("retry").hidden = true; $("error").textContent = "";
  try {state = await api("/api/game"); selection = null; render(); if (!state.result && state.turn !== state.human) state = await api("/api/ai", {});}
  catch (error) {$("error").textContent = `接続できませんでした。サーバーの起動を確認してください。${error.message}`; $("retry").hidden = false;}
  finally {busy = false; render();}
}
$("retry").onclick = connect;
connect();
