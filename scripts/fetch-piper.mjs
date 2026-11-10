// Fetches the pinned Piper release into the app's resource folder.
//
// Node 22+ built-in modules only: this runs as a Tauri before-build step, so it must
// never need an install.
//
// Usage: node scripts/fetch-piper.mjs
//
// Skips all network access when crates/regent-app/resources/piper/.fetched already
// carries the expected SHA-256 and the piper executable is present.

import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import {
  copyFileSync,
  createWriteStream,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const manifestPath = join(repoRoot, 'third_party', 'piper', 'manifest.json');
const resourceDir = join(repoRoot, 'crates', 'regent-app', 'resources', 'piper');
const stampPath = join(resourceDir, '.fetched');
const manifestLabel = 'third_party/piper/manifest.json';

class FetchError extends Error {}

function fail(message) {
  throw new FetchError(message);
}

function hostTriple() {
  const result = spawnSync('rustc', ['-vV'], { encoding: 'utf8' });
  if (result.error || result.status !== 0) {
    fail(`could not run \`rustc -vV\` to detect the target triple: ${result.error ?? result.stderr}`);
  }
  const line = result.stdout.split(/\r?\n/).find((l) => l.startsWith('host:'));
  if (!line) fail('could not find a `host:` line in `rustc -vV` output');
  return line.slice('host:'.length).trim();
}

function sha256File(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function copyDir(from, to) {
  mkdirSync(to, { recursive: true });
  for (const entry of readdirSync(from, { withFileTypes: true })) {
    const source = join(from, entry.name);
    const target = join(to, entry.name);
    if (entry.isDirectory()) {
      copyDir(source, target);
    } else if (entry.isFile()) {
      copyFileSync(source, target);
    }
  }
}

async function main() {
  const triple = process.env.TAURI_ENV_TARGET_TRIPLE || hostTriple();
  const exeSuffix = triple.includes('windows') ? '.exe' : '';
  const exeName = `piper${exeSuffix}`;

  if (!existsSync(manifestPath)) fail(`missing ${manifestLabel}`);

  let manifest;
  try {
    manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  } catch (error) {
    fail(`could not parse ${manifestLabel}: ${error.message}`);
  }

  const asset = manifest.assets?.[triple];
  if (!asset) fail(`no entry for target triple \`${triple}\` in ${manifestLabel}`);

  const { url, sha256, size } = asset;
  if (!url || !sha256 || sha256 === 'TODO' || !size) {
    fail(
      `the entry for \`${triple}\` in ${manifestLabel} is incomplete ` +
        '(url, sha256 and size are all required; fill them in with measured values)',
    );
  }

  // Already fetched: verify the stamp and the executable, then stop before touching
  // the network at all.
  if (
    existsSync(join(resourceDir, exeName)) &&
    existsSync(stampPath) &&
    readFileSync(stampPath, 'utf8').trim() === sha256
  ) {
    console.log(`Piper already present (${triple})`);
    return;
  }

  const workDir = mkdtempSync(join(tmpdir(), 'regent-piper-'));
  const archiveName = url.slice(url.lastIndexOf('/') + 1);
  const partPath = join(tmpdir(), `${archiveName}.part`);
  const extractDir = join(workDir, 'extract');
  let installed = false;

  try {
    console.log(`fetch-piper: downloading ${url}`);
    const response = await fetch(url, { redirect: 'follow' });
    if (!response.ok || !response.body) {
      fail(`download failed for ${url}: HTTP ${response.status}`);
    }
    await pipeline(Readable.fromWeb(response.body), createWriteStream(partPath));

    const actualSize = statSync(partPath).size;
    if (actualSize !== size) {
      fail(`size mismatch for ${archiveName}: expected ${size} bytes, got ${actualSize}`);
    }

    const actualSha = sha256File(partPath);
    if (actualSha !== sha256) {
      fail(`sha256 mismatch for ${archiveName}: expected ${sha256}, got ${actualSha}`);
    }

    mkdirSync(extractDir, { recursive: true });
    // The system `tar` handles .tar.gz everywhere and .zip on Windows and macOS.
    const tar = spawnSync('tar', ['-xf', partPath, '-C', extractDir], { encoding: 'utf8' });
    if (tar.error || tar.status !== 0) {
      fail(`tar failed to extract ${archiveName}: ${tar.error?.message ?? tar.stderr}`);
    }

    const extractedPiper = join(extractDir, 'piper');
    if (!existsSync(extractedPiper)) fail(`${archiveName} did not contain a piper/ folder`);
    if (!existsSync(join(extractedPiper, exeName))) {
      fail(`${archiveName} did not contain piper/${exeName} for ${triple}`);
    }

    mkdirSync(resourceDir, { recursive: true });
    copyDir(extractedPiper, resourceDir);
    writeFileSync(stampPath, `${sha256}\n`);
    installed = true;
    console.log(
      `fetch-piper: installed Piper ${manifest.version} into ${relative(repoRoot, resourceDir)}`,
    );
  } finally {
    rmSync(partPath, { force: true });
    rmSync(workDir, { recursive: true, force: true });
    // Leave nothing half-installed: the stamp is what marks a usable folder.
    if (!installed) rmSync(stampPath, { force: true });
  }
}

try {
  await main();
} catch (error) {
  console.error(`fetch-piper: ${error instanceof FetchError ? error.message : error.stack}`);
  process.exit(1);
}
