export const meta = {
  name: 'xyz2-consensus-write',
  description: 'Write the final XYZ^2 usefulness consensus document and verify the reviewers\' notes on stale repository docs',
  phases: [{ title: 'Write', detail: 'final document, merged doc notes, one verifier per doc note' }],
}
const IN = __INPUT__
const CACHE = IN.cached
const once = (label, prompt, opts) => (label in CACHE) ? Promise.resolve(CACHE[label]) : agent(prompt, { ...opts, label, phase: 'Write' })
const DOCFIX = {
  type: 'object',
  properties: { valid: { type: 'boolean' }, file: { type: 'string' }, old_text: { type: 'string' }, new_text: { type: 'string' }, reason: { type: 'string' } },
  required: ['valid', 'file', 'old_text', 'new_text', 'reason'],
}
phase('Write')
const docMerge = IN.docNotes.length ? once('docfix:merge',
  `Merge duplicates in these reviewer notes on stale repository documentation (repository /pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation) into distinct items, each with file, line and what is wrong. Drop notes about the consensus process itself.\n${IN.docNotes.join('\n')}`,
  { schema: { type: 'object', properties: { items: { type: 'array', items: { type: 'string' } } }, required: ['items'] } }) : Promise.resolve({ items: [] })
const [doc, docList] = await Promise.all([
  once('write:final',
    `${IN.brief2}\n\nWrite the final consensus document (markdown) on when the XYZ^2 code is useful, from this draft that ${IN.consensus ? 'reached' : 'did not fully reach'} consensus after ${IN.rounds} review rounds. Structure: a short answer first (plain statement, negative where the data are negative); then a table of regimes with verdict, the key numbers with uncertainties and the source file; then where XYZ^2 is useful or competitive, where it is not, what is untested and what would test it; then an 'Open disputes' section: for every item below 80% agreement in the last round, give the claim's current text in one line, the reviewers' objections and their proposed correction, and say whether the objection changes the verdict; then the material omissions the last round's reviewers raised, each in one line with its source file; do not silently adopt or drop any of these; then the agreement record (fraction of reviewers agreeing per claim in the last round, and the number of claims at >=80% in every round). Plain declarative sentences, no em dashes. No process history beyond the agreement record. Return only the markdown.\n\nDraft:\n${JSON.stringify(IN.draft)}\n\nObjections to the items below 80% in the last round:\n${JSON.stringify(IN.residual)}\n\nMaterial omissions raised in the last round:\n${JSON.stringify(IN.omissions)}\n\nAgreement history:\n${JSON.stringify(IN.history)}\n\n${IN.rules2}`,
    {}),
  docMerge,
])
const items = (docList && docList.items) || []
if (items.length > 40) log(`doc fixes: verifying the first 40 of ${items.length} items; the rest are returned unverified`)
const fixes = await pipeline(items.slice(0, 40), (it, _, k) => once(`docfix:${k}`,
  `${IN.brief2}\n\nVerify one reviewer note on stale repository documentation and, if it is right, give the exact edit.\n\nNote: ${it}\n\nRead the file and check the note against the data files. Return valid=false if the note is wrong. For a valid note give the file path, the exact current text to replace (verbatim and unique in the file) and the replacement text in the file's own style. Do not edit any file yourself.\n\n${IN.rules2}`,
  { schema: DOCFIX, effort: 'high' }))
return { ok: !!doc, failedDocfix: fixes.filter(f => !f).length, items: items.length }
