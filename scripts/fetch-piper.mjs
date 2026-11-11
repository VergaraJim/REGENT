// Fetches the pinned Piper release into the app's resource folder.
//
// Node 22+ built-in modules only: this runs as a Tauri before-build step, so it must
// never need an install.
//
// Usage: node scripts/fetch-piper.mjs
//
// Skips all network access when crates/regent-app/resources/piper/.fetched already
// carries the expected SHA-256 plus the current layout version and the piper executable
// is present.

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

// Bumped whenever the installed folder layout changes, so existing installs re-fetch
// even though their archive SHA-256 still matches.
// 2: symlinks in the archive are copied as their dereferenced regular files.
const LAYOUT_VERSION = 2;

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

// The dynamic Visual C++ runtime that piper.exe and its DLLs import. Microsoft allows
// these to be copied next to the executable that uses them (app-local deployment) from
// the Visual Studio `VC\Redist\MSVC` folder, so a clean Windows machine without the
// VC++ redistributable can still speak.
const vcRuntimeNames = ['msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'];

// Maps lower-case name -> actual entry name in `dir`, or null when `dir` is unreadable.
function filesByName(dir) {
  try {
    const map = new Map();
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.isFile()) map.set(entry.name.toLowerCase(), entry.name);
    }
    return map;
  } catch {
    return null;
  }
}

function vcRuntimeSource(dir) {
  const files = filesByName(dir);
  if (!files || !vcRuntimeNames.every((name) => files.has(name))) return null;
  return files;
}

// Folder names that are dotted integers (`14.44.35207`), as numbers, or null.
function versionSegments(name) {
  const segments = name.split('.');
  if (!segments.every((segment) => /^\d+$/.test(segment))) return null;
  return segments.map(Number);
}

function compareSegments(a, b) {
  for (let i = 0; i < Math.max(a.length, b.length); i += 1) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    if (diff !== 0) return diff;
  }
  return 0;
}

// Every integer group in a name, so `Microsoft.VC143.CRT` sorts by its 143.
function numericKey(name) {
  return (name.match(/\d+/g) ?? []).map(Number);
}

// The `x64\Microsoft.VC14*.CRT` folders of a `VC\Redist\MSVC` (or `VCToolsRedistDir`)
// root, highest version first.
function crtFolders(redistDir) {
  const x64 = join(redistDir, 'x64');
  let entries;
  try {
    entries = readdirSync(x64, { withFileTypes: true });
  } catch {
    return [];
  }
  return entries
    .filter((entry) => entry.isDirectory() && /^Microsoft\.VC14.*\.CRT$/i.test(entry.name))
    .map((entry) => join(x64, entry.name))
    .sort((a, b) => compareSegments(numericKey(b), numericKey(a)));
}

function findVcRuntimeDir() {
  const override = process.env.REGENT_VC_REDIST_DIR;
  if (override) {
    // An explicit override is authoritative: a wrong path is a mistake worth reporting,
    // not something to paper over with a different source.
    const files = vcRuntimeSource(override);
    if (!files) failVcRuntime();
    return { dir: override, files };
  }

  const toolsRedist = process.env.VCToolsRedistDir;
  if (toolsRedist) {
    for (const dir of crtFolders(toolsRedist)) {
      const files = vcRuntimeSource(dir);
      if (files) return { dir, files };
    }
  }

  const programFilesX86 = process.env['ProgramFiles(x86)'] || process.env.ProgramFiles;
  if (programFilesX86) {
    const vswhere = join(programFilesX86, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe');
    if (existsSync(vswhere)) {
      const result = spawnSync(
        vswhere,
        [
          '-latest',
          '-products',
          '*',
          '-requires',
          'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
          '-property',
          'installationPath',
        ],
        { encoding: 'utf8' },
      );
      const installationPath = result.status === 0 ? result.stdout.trim() : '';
      if (installationPath) {
        const redistRoot = join(installationPath, 'VC', 'Redist', 'MSVC');
        let versions;
        try {
          versions = readdirSync(redistRoot, { withFileTypes: true });
        } catch {
          versions = [];
        }
        const ordered = versions
          .filter((entry) => entry.isDirectory() && versionSegments(entry.name) !== null)
          .map((entry) => entry.name)
          .sort((a, b) => compareSegments(versionSegments(b), versionSegments(a)));
        for (const version of ordered) {
          for (const dir of crtFolders(join(redistRoot, version))) {
            const files = vcRuntimeSource(dir);
            if (files) return { dir, files };
          }
        }
      }
    }
  }

  failVcRuntime();
}

function failVcRuntime() {
  fail(
    `could not find ${vcRuntimeNames.join(', ')} in a Visual Studio redistributable folder; ` +
      'set REGENT_VC_REDIST_DIR to a folder holding those DLLs, or install the ' +
      'Visual Studio C++ build tools',
  );
}

// The stamp records both the archive SHA-256 and the layout it was copied with.
function stampValue(sha256) {
  return `${sha256} layout-${LAYOUT_VERSION}`;
}

function ensureVcRuntime(triple) {
  if (!triple.includes('windows')) return;

  const { dir, files } = findVcRuntimeDir();
  let copied = false;
  for (const name of vcRuntimeNames) {
    const source = join(dir, files.get(name));
    const target = join(resourceDir, name);
    if (existsSync(target)) {
      const existing = statSync(target);
      const incoming = statSync(source);
      if (existing.size === incoming.size && sha256File(target) === sha256File(source)) continue;
    }
    copyFileSync(source, target);
    copied = true;
  }
  console.log(
    copied
      ? `fetch-piper: VC++ runtime copied from ${dir}`
      : 'fetch-piper: VC++ runtime already present',
  );
}

// Copies `from` into `to`, dereferencing symlinks: the Linux Piper archive ships its
// libraries as a real versioned file plus links the executable loads by name. Copying
// the link targets as regular files keeps those names working, and regular files also
// survive Tauri bundling and Windows checkouts.
function copyDir(from, to) {
  mkdirSync(to, { recursive: true });
  for (const entry of readdirSync(from, { withFileTypes: true })) {
    const source = join(from, entry.name);
    const target = join(to, entry.name);
    if (entry.isDirectory()) {
      copyDir(source, target);
    } else if (entry.isFile()) {
      copyFileSync(source, target);
    } else if (entry.isSymbolicLink()) {
      let stat;
      try {
        stat = statSync(source);
      } catch {
        console.warn(`fetch-piper: skipping dangling link ${relative(repoRoot, source)}`);
        continue;
      }
      if (stat.isDirectory()) {
        copyDir(source, target);
      } else if (stat.isFile()) {
        copyFileSync(source, target);
      }
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
    readFileSync(stampPath, 'utf8').trim() === stampValue(sha256)
  ) {
    console.log(`Piper already present (${triple})`);
    ensureVcRuntime(triple);
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
    writeFileSync(stampPath, `${stampValue(sha256)}\n`);
    installed = true;
    console.log(
      `fetch-piper: installed Piper ${manifest.version} into ${relative(repoRoot, resourceDir)}`,
    );
    ensureVcRuntime(triple);
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
