export const meta = {
  name: 'xyz2-consensus-review',
  description: 'Rewrite the overall answer of the XYZ^2 usefulness consensus, then a ten-reviewer vote and an omission triage',
  phases: [
    { title: 'Overall', detail: 'rewrite the overall answer from the revised claims' },
    { title: 'Review', detail: '10 lens-diverse reviewers vote on every claim and the overall answer' },
    { title: 'Omissions', detail: 'merge reviewer omissions and judge materiality' },
  ],
}
const IN = __INPUT__
const CACHE = IN.cached
const once = (label, prompt, opts) => (label in CACHE) ? Promise.resolve(CACHE[label]) : agent(prompt, { ...opts, label })
const OVERALL = {
  type: 'object',
  properties: { overall: { type: 'string' }, open_questions: { type: 'array', items: { type: 'string' } } },
  required: ['overall', 'open_questions'],
}
const REVIEW2 = {
  type: 'object',
  properties: {
    reviews: { type: 'array', items: { type: 'object', properties: {
      claim_id: { type: 'string' }, agree: { type: 'boolean' },
      severity: { type: 'string', enum: ['none', 'minor', 'major'] },
      correction: { type: 'string' }, evidence: { type: 'string' } },
      required: ['claim_id', 'agree', 'severity', 'correction', 'evidence'] } },
    missing: { type: 'array', items: { type: 'string' }, description: 'regimes, evidence or caveats that would change a verdict or a headline number' },
    doc_fixes: { type: 'array', items: { type: 'string' }, description: 'stale repository documentation: file, line, what is wrong, evidence' },
    overall_agree: { type: 'boolean' },
  },
  required: ['reviews', 'missing', 'doc_fixes', 'overall_agree'],
}
const TRIAGE = {
  type: 'object',
  properties: { items: { type: 'array', items: { type: 'object', properties: {
    text: { type: 'string' }, scope: { type: 'string', enum: ['out', 'doc', 'substantive'] },
    material: { type: 'boolean' }, claim_ids: { type: 'array', items: { type: 'string' } }, reason: { type: 'string' } },
    required: ['text', 'scope', 'material', 'claim_ids', 'reason'] } } },
  required: ['items'],
}
let draft = IN.draft
const VERIFY = {
  type: 'object',
  properties: { accept: { type: 'boolean' }, problems: { type: 'array', items: { type: 'string' } } },
  required: ['accept', 'problems'],
}
if (IN.overallRound) {
  phase('Overall')
  const N = IN.overallRound
  const writePrompt = `${IN.brief2}\n\nRewrite the overall answer and the open questions of the consensus on where the XYZ^2 code is useful, from the claims below. The overall answer comes first and is plain: where XYZ^2 is useful, competitive or not, negative where the data are negative, with the key numbers and the claim ids. It is a summary of the claims: every sentence must be traceable to a claim id, and it must agree with every claim. Do not add numbers that are not in the claims. Use 'every', 'no' or 'never' only where a claim states it, and name every exception the claims name. Use the conventions in the brief. No process history. Plain declarative sentences, no em dashes.\n\nReviewer notes on the previous overall answer:\n${JSON.stringify(IN.overallNotes)}\n\nPrevious overall answer:\n${draft.overall}\n\nPrevious open questions:\n${JSON.stringify(draft.open_questions)}\n\nClaims:\n${JSON.stringify(draft.claims)}\n\n${IN.rules2}`
  let ov = await once(`overall:r${N}`, writePrompt, { phase: 'Overall', schema: OVERALL })
  if (!ov) return { failed: 'overall' }
  if (IN.verifyOverall) {
    const lenses = ['consistency: every sentence of the overall answer against the claims it cites, every exception the claims name, and every quantifier (every, no, never, only)',
      'numbers and significance: every number and significance statement in the overall answer against the claims and the files']
    const vs = await parallel(lenses.map((lens, k) => () => once(`overall-verify:r${N}:${k}`,
      `${IN.brief2}\n\nAdversarially verify the overall answer of the consensus on where the XYZ^2 code is useful. Your focus: ${lens}.\n\nReviewer notes on the previous overall answer:\n${JSON.stringify(IN.overallNotes)}\n\nNew overall answer and open questions:\n${JSON.stringify(ov)}\n\nClaims:\n${JSON.stringify(draft.claims)}\n\nReturn accept=false only with concrete problems and evidence (claim ids, numbers, file paths).\n\n${IN.rules2}`,
      { phase: 'Overall', schema: VERIFY, effort: 'high' })))
    if (vs.some(v => !v)) return { failed: 'overall-verify' }
    const problems = vs.filter(v => !v.accept).flatMap(v => v.problems)
    if (problems.length) {
      const g = await once(`overall-refix:r${N}`, `${writePrompt}\n\nYour previous version:\n${JSON.stringify(ov)}\n\nVerifiers found these problems:\n${problems.join('\n')}\n\nReturn a corrected version that fixes the problems the claims and files confirm.`, { phase: 'Overall', schema: OVERALL })
      if (!g) return { failed: 'overall-refix' }
      ov = g
    }
  }
  draft = { ...draft, overall: ov.overall, open_questions: ov.open_questions }
}
const R = IN.round
const ids = draft.claims.map(c => c.id)
const index = draft.claims.map(c => `${c.id}: ${c.regime}`).join('\n')
phase('Review')
const reviews = await parallel(IN.reviewers.map((lens, i) => () => once(`review:r${R}:${i}`,
  `${IN.brief2}\n\nYou are reviewer ${i + 1} of ${IN.reviewers.length} in review round ${R} of a consensus on where the XYZ^2 code is useful. Your lens: ${lens}.\n\nVote on every claim (${ids.join(', ')}) and on the overall answer (id OVERALL). Agreement standard: agree=true when the verdict and the statement are correct and every number matches the cited files within rounding. Minor wording or precision suggestions go in 'correction' with severity 'minor' and agree=true. Use agree=false with severity 'major' when a verdict is wrong, a number is wrong or not in the files, a claim is stronger or weaker than the data, or a significance statement fails your check. Be adversarial on substance: check the numbers yourself and default to disagree when a claim is not backed by the files. Do not disagree over style alone.\n\nChanges since the previous round (check that each fix is right):\n${IN.changelog}\n\n'missing' is only for regimes, evidence or caveats that would change a verdict or a headline number. Report stale repository documentation in 'doc_fixes', not in 'missing'.\n\nDraft (round ${R}):\n${JSON.stringify(draft)}\n\n${IN.rules2}`,
  { phase: 'Review', schema: REVIEW2 })))
const done = reviews.filter(Boolean)
if (done.length < reviews.length) return { failed: 'review', done: done.length }
const agreeOf = id => { const v = done.map(r => r.reviews.find(x => x.claim_id === id)).filter(Boolean); return v.length ? v.filter(x => x.agree).length / v.length : 0 }
const weak = [...ids, 'OVERALL'].filter(id => agreeOf(id) < 0.8)
log(`round ${R}: ${ids.length + 1 - weak.length}/${ids.length + 1} items at >=80% agreement${weak.length ? `; weak: ${weak.join(', ')}` : ''}`)
phase('Omissions')
const missing = reviews.flatMap((r, i) => (r.missing || []).map(m => `[reviewer ${i + 1}] ${m}`))
const triage = missing.length ? await once(`triage:r${R}`,
  `${IN.brief2}\n\nReviewers in round ${R} of the consensus on where the XYZ^2 code is useful listed these omissions:\n${missing.join('\n')}\n\nClaim index:\n${index}\n\nMerge duplicates into distinct items. For each item set scope: 'out' (review-process history, the session, commits, style), 'doc' (stale repository documentation outside the consensus) or 'substantive'. For substantive items check the files and set material=true only if the item would change a verdict, a headline number or a caveat a reader needs. Give the claim ids it belongs to, or an empty list if it needs a new claim.\n\n${IN.rules2}`,
  { phase: 'Omissions', schema: TRIAGE }) : { items: [] }
if (!triage) return { failed: 'triage' }
const material = triage.items.filter(x => x.scope === 'substantive' && x.material).length
log(`round ${R}: ${material} material omissions`)
return { ok: true, weak, material }
