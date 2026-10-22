export const meta = {
  name: 'xyz2-consensus-fix',
  description: 'Revise the disputed claims of the XYZ^2 usefulness consensus; independent verifiers check each revision',
  phases: [{ title: 'Fix', detail: 'one fixer per disputed claim, then verifiers, then a refix when they find problems' }],
}
const IN = __INPUT__
const CACHE = IN.cached
const R = IN.round
const once = (label, prompt, opts) => (label in CACHE) ? Promise.resolve(CACHE[label]) : agent(prompt, { ...opts, label, phase: 'Fix' })
const FIX = {
  type: 'object',
  properties: { claim: IN.claimSchema, drop: { type: 'boolean' }, changes: { type: 'array', items: { type: 'string' } }, rebuttals: { type: 'array', items: { type: 'string' } } },
  required: ['claim', 'drop', 'changes', 'rebuttals'],
}
const VERIFY = {
  type: 'object',
  properties: { accept: { type: 'boolean' }, problems: { type: 'array', items: { type: 'string' } } },
  required: ['accept', 'problems'],
}
const NEWCLAIM = {
  type: 'object',
  properties: { keep: { type: 'boolean' }, reason: { type: 'string' }, claim: IN.claimSchema },
  required: ['keep', 'reason'],
}
const fixPrompt = t => `${IN.brief2}\n\nYou revise one claim of the consensus draft on where the XYZ^2 code is useful. Agreement in review round ${R}: ${Math.round(t.frac * 100)}% of reviewers.\n\nClaim:\n${JSON.stringify(t.claim)}\n\nReviewer notes (agree=false are objections):\n${JSON.stringify(t.notes)}\n\n${t.omissions.length ? `Material omissions assigned to this claim:\n${t.omissions.join('\n')}\n\n` : ''}Other claims, for cross-references:\n${IN.index}\n\nCheck every objection and omission against the files yourself. ${t.frac >= 0.8 && !t.omissions.length ? 'This claim already has at least 80% agreement: change only what the files force (a wrong number, significance or verdict) and add nothing else.' : 'Fix every objection and omission the files confirm.'} Answer objections the files refute in 'rebuttals' with the evidence, and leave that text unchanged. Prefer removing a disputed detail the verdict does not need over adding new detail. Set drop=true only if the files show the claim cannot stand in any form. Keep the id. No revision history in the claim text. Plain declarative sentences, no em dashes. Quote numbers with uncertainties and full file paths. List each change in 'changes'.\n\n${IN.rules2}`
const verifyPrompt = (t, f, lens) => `${IN.brief2}\n\nAdversarially verify a revised claim of the consensus on where the XYZ^2 code is useful. Your focus: ${lens}.\n\nOriginal claim:\n${JSON.stringify(t.claim)}\n\nReviewer notes:\n${JSON.stringify(t.notes)}\n\n${t.omissions.length ? `Material omissions assigned to this claim:\n${t.omissions.join('\n')}\n\n` : ''}Revision (drop=true means the fixer proposes removing the claim):\n${JSON.stringify(f)}\n\nCheck against the files: every objection the files confirm is fixed, every rebuttal is right, every number in the revised claim is correct and sourced, and the verdict follows. Return accept=false only with concrete problems and evidence (numbers, file paths).\n\n${IN.rules2}`
const newPrompt = t => `${IN.brief2}\n\nReviewers flagged an omission in the consensus on where the XYZ^2 code is useful:\n${t.text}\n\nTriage reason: ${t.reason}\n\nClaim index:\n${IN.index}\n\nCheck the files. If the omission is real and material and no existing claim covers it, write a new claim with id ${t.id} and keep=true. Otherwise keep=false with the reason. Plain declarative sentences, no em dashes. Numbers with uncertainties and full file paths.\n\n${IN.rules2}`
const newVerifyPrompt = (t, n, lens) => `${IN.brief2}\n\nAdversarially verify a proposed new claim for the consensus on where the XYZ^2 code is useful. Your focus: ${lens}.\n\nThe omission it answers:\n${t.text}\n\nProposed claim:\n${JSON.stringify(n.claim)}\n\nExisting claims:\n${IN.index}\n\nAccept only if it is correct, material and not already covered. Return accept=false with concrete problems otherwise.\n\n${IN.rules2}`

const results = await pipeline(IN.tasks,
  t => t.kind === 'new'
    ? once(`new:r${R}:${t.id}`, newPrompt(t), { schema: NEWCLAIM })
    : once(`fix:r${R}:${t.id}`, fixPrompt(t), t.light ? { schema: FIX, effort: 'high' } : { schema: FIX }),
  (f, t) => {
    if (!f) return { id: t.id, failed: 'first' }
    if (t.kind === 'new') {
      if (!f.keep || !f.claim) return { id: t.id, done: true }
      return parallel(IN.lenses.map((lens, k) => () => once(`verify:r${R}:${t.id}:${k}`, newVerifyPrompt(t, f, lens), { schema: VERIFY, effort: 'high' })))
        .then(vs => vs.some(v => !v) ? { id: t.id, failed: 'verify' } : { id: t.id, done: true })
    }
    if (t.light && !f.drop && !(f.changes || []).length) return { id: t.id, done: true }
    const n = t.light ? 1 : 2
    return parallel(IN.lenses.slice(0, n).map((lens, k) => () => once(`verify:r${R}:${t.id}:${k}`, verifyPrompt(t, f, lens), { schema: VERIFY, effort: 'high' })))
      .then(vs => ({ id: t.id, f, vs }))
  },
  (x, t) => {
    if (!x || x.failed || x.done) return x
    if (x.vs.some(v => !v)) return { id: t.id, failed: 'verify' }
    const problems = x.vs.filter(v => !v.accept).flatMap(v => v.problems)
    if (!problems.length) return { id: t.id, done: true }
    return once(`refix:r${R}:${t.id}`, `${fixPrompt(t)}\n\nYour previous revision:\n${JSON.stringify(x.f)}\n\nVerifiers found these problems:\n${problems.join('\n')}\n\nReturn a corrected revision that fixes the problems the files confirm.`, { schema: FIX })
      .then(g => g ? { id: t.id, done: true } : { id: t.id, failed: 'refix' })
  })
const failed = results.filter(r => !r || r.failed)
log(`round ${R} fixes: ${results.length - failed.length}/${IN.tasks.length} done${failed.length ? `; failed: ${failed.map(r => r ? `${r.id} (${r.failed})` : '?').join(', ')}` : ''}`)
return { round: R, done: results.length - failed.length, failed: failed.map(r => r && r.id) }
