// Renderiza a linha do tempo de index.html quadro a quadro com Playwright.
// Uso:
//   node render.mjs frames <w> <h> <saida_dir> <t1,t2,...>   -> PNGs avulsos (folha de contato)
//   node render.mjs video  <w> <h> <saida.mp4> [fps] [audio.wav]
//   node render.mjs events <saida.json>                       -> eventos para a trilha
import { createRequire } from 'module';
import { spawn } from 'child_process';
import { mkdirSync, writeFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const require = createRequire(import.meta.url);
let pw;
try { pw = require('playwright'); } catch { pw = require('/opt/node22/lib/node_modules/playwright'); }
const here = path.dirname(fileURLToPath(import.meta.url));
const [mode, ...args] = process.argv.slice(2);
const DUR = 20;

async function open(w, h) {
  const browser = await pw.chromium.launch({ args: ['--allow-file-access-from-files', '--force-color-profile=srgb'] });
  const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
  await page.goto('file://' + path.join(here, 'index.html') + `?w=${w}&h=${h}`);
  await page.evaluate(() => window.READY);
  return { browser, page };
}

if (mode === 'events') {
  const { browser, page } = await open(1920, 1080);
  writeFileSync(args[0], JSON.stringify(await page.evaluate(() => window.EVENTS), null, 1));
  await browser.close();
} else if (mode === 'frames') {
  const [w, h, dir, list] = args;
  mkdirSync(dir, { recursive: true });
  const { browser, page } = await open(+w, +h);
  for (const t of list.split(',').map(Number)) {
    await page.evaluate(t => window.seek(t), t);
    await page.screenshot({ path: path.join(dir, `f_${t.toFixed(2)}.png`) });
  }
  await browser.close();
} else if (mode === 'video') {
  const [w, h, out, fpsArg, audio] = args;
  const fps = +(fpsArg || 60);
  if (existsSync(out)) { console.error('Arquivo já existe, não sobrescrevo:', out); process.exit(1); }
  const { browser, page } = await open(+w, +h);
  const ff = ['-hide_banner', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'png', '-i', '-'];
  if (audio) ff.push('-i', audio, '-c:a', 'aac', '-b:a', '320k', '-shortest');
  ff.push('-c:v', 'libx264', '-preset', 'slow', '-crf', '15', '-pix_fmt', 'yuv420p', '-color_primaries', 'bt709',
    '-color_trc', 'bt709', '-colorspace', 'bt709', '-movflags', '+faststart', '-n', out);
  const enc = spawn('ffmpeg', ff, { stdio: ['pipe', 'inherit', 'inherit'] });
  const N = Math.round(DUR * fps);
  const t0 = Date.now();
  for (let i = 0; i < N; i++) {
    await page.evaluate(t => window.seek(t), i / fps);
    const buf = await page.screenshot({ type: 'png' });
    if (!enc.stdin.write(buf)) await new Promise(r => enc.stdin.once('drain', r));
    if (i % 120 === 0) console.log(`${w}x${h} quadro ${i}/${N} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  }
  enc.stdin.end();
  await new Promise((res, rej) => enc.on('close', c => c === 0 ? res() : rej(new Error('ffmpeg ' + c))));
  await browser.close();
  console.log('ok', out);
}
