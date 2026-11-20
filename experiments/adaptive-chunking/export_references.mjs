// Resolve module-owned 5etools references with its native copy/merge logic.
// Files are read locally; only the requested experiment output is written.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';

const [repository, sourceFile, destination] = process.argv.slice(2);
if (!repository || !sourceFile || !destination) throw new Error('Usage: repository source-json output-json');
const repo = path.resolve(repository), output = path.resolve(destination), source = path.resolve(sourceFile);
process.chdir(repo);
for (const name of ['js/parser.js', 'js/utils.js', 'js/render.js'])
  await import(pathToFileURL(path.join(repo, name)).href);
const utils = await import(pathToFileURL(path.join(repo, 'node/util.js')).href);
utils.patchLoadJson();
const inputs = new Map();
for (const method of ['loadJSON', 'loadRawJSON']) {
  const original = DataUtil[method].bind(DataUtil);
  DataUtil[method] = async function (url, ...args) {
    if (/^https?:/i.test(url)) throw new Error('Network access is disabled for this export');
    const file = path.resolve(repo, url);
    if (!file.startsWith(repo + path.sep)) throw new Error('Reference escapes the source repository');
    if (fs.existsSync(file)) inputs.set(file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'));
    return original(url, ...args);
  };
}
const adventure = JSON.parse(fs.readFileSync(source, 'utf8'));
const requests = [];
function visit(value) {
  if (Array.isArray(value)) value.forEach(visit);
  else if (value && typeof value === 'object') {
    if (value.type === 'statblock') requests.push(value);
    Object.values(value).forEach(visit);
  }
}
visit(adventure);
const directRequestCount = requests.length;
const sourceId = adventure.adventureData[0].source;
const pools = {};
async function getPool(kind) {
  if (pools[kind]) return pools[kind];
  const suffix = sourceId.toLowerCase();
  const routes = {
    creature: [`data/bestiary/bestiary-${suffix}.json`, 'monster'],
    monsterFluff: [`data/bestiary/fluff-bestiary-${suffix}.json`, 'monsterFluff'],
    item: ['data/items.json', 'item'], reward: ['data/rewards.json', 'reward'],
  };
  if (!routes[kind]) throw new Error(`Unsupported reference kind ${kind}`);
  const [file, collection] = routes[kind];
  const data = await DataUtil.loadJSON(file);
  pools[kind] = [...(data[collection] || []), ...(kind === 'item' ? data.itemGroup || [] : [])];
  return pools[kind];
}
const resolved = {};
for (const ref of requests) {
  const kind = ref.tag || ref.prop;
  const key = [kind, ref.source || sourceId, ref.name].join('|');
  if (resolved[key]) continue;
  const pool = await getPool(kind);
  const entity = pool.find(x => x.name === ref.name && x.source === (ref.source || sourceId));
  if (!entity) throw new Error(`Unresolved module reference ${key}`);
  if (entity._copy) throw new Error(`Unmerged entity ${key}`);
  resolved[key] = entity;
  if (kind === 'item' && Array.isArray(entity.items)) {
    for (const uid of entity.items) {
      if (typeof uid !== 'string') continue;
      const [name, itemSource = 'DMG'] = uid.split('|');
      if (itemSource === sourceId) requests.push({tag: 'item', name, source: itemSource});
    }
  }
}
const result = {module_source: sourceId, requested_references: directRequestCount,
  references: resolved, source_files: Object.fromEntries(inputs)};
fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n', {flag: 'wx'});
console.log(JSON.stringify({requested: directRequestCount, resolved: Object.keys(resolved).length, source_files: inputs.size}));
