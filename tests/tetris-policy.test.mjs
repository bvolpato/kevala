import assert from "node:assert/strict";
import test from "node:test";

import { AutoPlayer, candidates, describe, summary } from "../app/tetris/ai.js";
import { Game, H, W, dropDistance, measure, newBoard, spawnPiece } from "../app/tetris/engine.js";

test("board descriptions report existing holes even when a move adds none", () => {
  const board = newBoard();
  board[(H - 2) * W] = 1;
  const spawn = { ...spawnPiece("O"), x: W - 2 };
  const landing = { ...spawn, y: spawn.y + dropDistance(board, spawn) };
  const outcome = measure(board, landing);

  assert.equal(outcome.holesBefore, 1);
  assert.equal(outcome.newHoles, 0);
  assert.equal(outcome.holesAfter, 1);
  assert.match(describe(outcome, [outcome]), /stack has one hole after this move/);
  assert.match(summary(outcome, [outcome]), /1 hole/);
});

test("safe landings exclude hole-making states from the model shortlist", () => {
  const game = new Game(1);
  game.hardDrop();
  const choices = candidates(game.board, game.piece);

  assert.ok(choices.items.some((item) => item.m.newHoles > 0));
  assert.ok(choices.texts.length > 0);
  assert.ok(choices.texts.length < choices.items.length);
  for (const group of choices.groups.values()) {
    assert.equal(group[0].m.newHoles, 0);
  }
});

test("a blocked planned key requests a reachable landing from the current pose", async () => {
  const game = new Game(1);
  let calls = 0;
  const model = {
    decideMany: async (items) => {
      calls++;
      return items.map(() => ({ answers: { clean: { noul: 0.8 } } }));
    },
  };
  const player = new AutoPlayer(game, { getModel: () => model });
  player.setEnabled(true);
  await new Promise((resolve) => setImmediate(resolve));
  assert.ok(player.plan);

  const previousCalls = calls;
  player.plan.spot.target.rot = (game.piece.rot + 1) % 4;
  game.rotate = () => false;
  player.tick(1000);

  assert.equal(player.plan, null);
  assert.ok(calls > previousCalls);
  await new Promise((resolve) => setImmediate(resolve));
  assert.ok(player.plan);
  player.setEnabled(false);
});
