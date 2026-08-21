/** Exercises Features + Model together, without a DOM. */
import fs from 'node:fs'; import path from 'node:path'; import vm from 'node:vm';

const root = path.resolve(import.meta.dirname, '..');
const ctx = {
  performance: { now: () => 0 }, console,
  document: { createElement: () => ({ set textContent(v){this._v=v;}, get innerHTML(){return this._v;} }) },
  fetch: async () => ({ ok: true, json: async () =>
    JSON.parse(fs.readFileSync(path.join(root,'model','model.json'),'utf8')) }),
};
vm.createContext(ctx);
for (const [file, name] of [['model.js','Model'], ['features.js','Features']]) {
  vm.runInContext(fs.readFileSync(path.join(root,'js',file),'utf8')
    + `\n;globalThis.${name} = ${name};`, ctx);
}
const { Model, Features } = ctx;
await Model.load('model/model.json');

let failed = 0;
const check = (name, cond, extra='') => {
  console.log(`${cond ? '  ok  ' : '  FAIL'} ${name}${extra ? '  ' + extra : ''}`);
  if (!cond) failed++;
};

console.log('\nuser id validation');
check('rejects empty',        !Features.validateUserId('').ok);
check('rejects spaces',       !Features.validateUserId('has space').ok);
check('rejects 31 chars',     !Features.validateUserId('x'.repeat(31)).ok);
check('rejects emoji',        !Features.validateUserId('user🌴').ok);
check('accepts dots/unders',   Features.validateUserId('nat.geo_1').ok);
check('strips leading @',      Features.validateUserId('@natgeo').value === 'natgeo');

console.log('\ndemo profile');
const a = Features.demoProfile('natgeo'), b = Features.demoProfile('natgeo');
check('deterministic', JSON.stringify(a) === JSON.stringify(b));
check('differs per user', JSON.stringify(a) !== JSON.stringify(Features.demoProfile('other')));
check('all 9 columns', Object.keys(a).length === 9, Object.keys(a).length + ' keys');
check('engagement in range', a['Engagement Rate (%)'] >= 0 && a['Engagement Rate (%)'] <= 100);

console.log('\nprediction');
const p = Model.predict(a);
check('label is a known class', Model.classes.includes(p.label), p.label);
check('confidence 0..1', p.confidence >= 0 && p.confidence <= 1, p.confidence.toFixed(4));
check('probabilities sum to 1',
  Math.abs(Object.values(p.probabilities).reduce((s,v)=>s+v,0) - 1) < 1e-9);
check('confidence === max prob', p.probabilities[p.label] === p.confidence);

console.log('\nrobustness (the old Flask app crashed on all of these)');
for (const bio of ['brand new never-seen text', '<script>alert(1)</script>', '🌴🔥💪',
                   '', 'x'.repeat(5000), "'; DROP TABLE users; --", 'Café naïve résumé']) {
  let ok = true; try { Model.predict({ ...a, 'Bio Text': bio }); } catch { ok = false; }
  check(`bio: ${JSON.stringify(bio.slice(0,26))}`, ok);
}
for (const [k,v] of [['Followers',0],['Followers',1e10],['Engagement Rate (%)',100],
                     ['Account Age (Years)',0],['Avg Likes per Post',1e9]]) {
  let ok = true; try { Model.predict({ ...a, [k]: v }); } catch { ok = false; }
  check(`extreme ${k}=${v}`, ok);
}
let nanOk = true; try { Model.predict({ ...a, Followers: NaN }); } catch { nanOk = false; }
check('NaN input does not throw', nanOk);

console.log('\nexplanation');
const ex = Model.explain(a, p.label);
check('one entry per input field', ex.length === 9, ex.length + ' entries');
check('sorted by |delta| desc',
  ex.every((r,i) => i === 0 || Math.abs(ex[i-1].delta) >= Math.abs(r.delta)));
check('bio is explained', ex.some(r => r.feature === 'Bio Text'));
check('deltas are finite', ex.every(r => Number.isFinite(r.delta)));

console.log('\nform validation');
const bad = Features.readForm({followers:'-5',following:'10',posts:'5',engagement:'200',
  likes:'1',comments:'1',age:'1',bio:'',verified:false});
check('rejects negative followers', !!bad.errors.followers);
check('rejects engagement > 100',   !!bad.errors.engagement);
check('rejects empty bio',          !!bad.errors.bio);
const good = Features.readForm({followers:'5000',following:'300',posts:'150',engagement:'4.5',
  likes:'400',comments:'20',age:'5',bio:'hello',verified:true});
check('accepts valid form', good.ok);
check('verified maps to 1', good.ok && good.profile['Verified'] === 1);

console.log(failed ? `\n${failed} CHECK(S) FAILED` : '\nALL LOGIC CHECKS PASSED');
process.exit(failed ? 1 : 0);
