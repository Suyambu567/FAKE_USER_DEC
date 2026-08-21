/** End-to-end check in a real browser: does the page render and work? */
// playwright ships CommonJS, so pull chromium off the default export.
// Resolved by name, not by absolute path: the path this used to hardcode only
// existed on one machine, so the test could not run anywhere else. Set
// NODE_PATH if playwright lives outside the usual resolution roots.
import playwright from 'playwright';
const { chromium } = playwright;

const BASE = process.argv[2] || 'http://127.0.0.1:8451';

// Let playwright find its own browser. Only override when the installed package
// cannot see the build it wants:
//   CHROME_PATH=/path/to/chrome node test/e2e.mjs
const executablePath = process.env.CHROME_PATH || undefined;

const browser = await chromium.launch({
  ...(executablePath ? { executablePath } : {}),
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});
let failed = 0;
const check = (name, cond, extra = '') => {
  console.log(`${cond ? '  ok  ' : '  FAIL'} ${name}${extra ? '  ' + extra : ''}`);
  if (!cond) failed++;
};

for (const [device, width, height] of [['mobile', 390, 844], ['desktop', 1280, 900]]) {
  console.log(`\n=== ${device} ${width}x${height} ===`);
  const page = await browser.newPage({ viewport: { width, height } });

  const consoleErrors = [];
  page.on('console', (m) => m.type() === 'error' && consoleErrors.push(m.text()));
  page.on('pageerror', (e) => consoleErrors.push(String(e)));

  await page.goto(BASE, { waitUntil: 'networkidle' });

  check('model loaded (status banner hidden)',
        await page.locator('#modelStatus').isHidden());
  check('model card visible', await page.locator('#modelCard').isVisible());

  // The honest-metrics requirement: baseline must appear next to accuracy.
  const metaText = await page.locator('#modelMeta').innerText();
  check('accuracy shown', /Accuracy/.test(metaText));
  check('baseline shown next to it', /baseline/i.test(metaText));
  check('synthetic-data warning surfaced',
        /synthetic/i.test(await page.locator('#modelWarnings').innerText()));

  // --- demo path (user ID only) ---
  await page.fill('#userId', 'natgeo');
  await page.click('#analyzeBtn');
  await page.waitForSelector('#result:not([hidden])', { timeout: 8000 });

  check('verdict rendered',
        ['Real User', 'Fake User'].includes(await page.locator('#verdictLabel').innerText()),
        await page.locator('#verdictLabel').innerText());
  check('demo banner shown for ID-only input',
        await page.locator('#demoBanner').isVisible());
  const conf = await page.locator('#confidenceValue').innerText();
  check('confidence is a percentage', /^\d+\.\d%$/.test(conf), conf);
  check('reasons listed', (await page.locator('#reasons li').count()) > 0,
        `${await page.locator('#reasons li').count()} items`);

  // --- real path (manual stats) ---
  await page.click('#profileDetails summary');
  for (const [id, v] of [['followers','120'],['following','7400'],['posts','3'],
                         ['engagement','0.4'],['likes','1'],['comments','0'],['age','0.1']]) {
    await page.fill(`#${id}`, v);
  }
  await page.fill('#bio', 'FREE FOLLOWERS >>> link below');
  await page.click('#analyzeBtn');
  await page.waitForTimeout(700);

  check('demo banner hidden when stats supplied',
        await page.locator('#demoBanner').isHidden());
  const label = await page.locator('#verdictLabel').innerText();
  check('bot-like profile classified Fake', label === 'Fake User', label);

  // --- validation ---
  await page.fill('#userId', 'bad name!!');
  await page.click('#analyzeBtn');
  await page.waitForTimeout(250);
  check('invalid user ID shows an inline error',
        await page.locator('#userIdError').isVisible());

  await page.fill('#userId', 'ok_user');
  await page.fill('#followers', '-5');
  await page.click('#analyzeBtn');
  await page.waitForTimeout(250);
  check('negative followers shows a field error',
        await page.locator('[data-error-for="followers"]').isVisible());

  // --- layout ---
  const overflow = await page.evaluate(() =>
    document.documentElement.scrollWidth - document.documentElement.clientWidth);
  check('no horizontal overflow', overflow <= 0, `${overflow}px`);

  // --- theme ---
  await page.click('#themeToggle');
  await page.waitForTimeout(150);
  check('theme toggle sets data-theme',
        ['dark', 'light'].includes(
          await page.evaluate(() => document.documentElement.getAttribute('data-theme'))));

  check('no console errors', consoleErrors.length === 0, consoleErrors.join(' | ').slice(0, 140));

  await page.screenshot({ path: `/tmp/claude-1000/-home-leelai-suyambu-rd-test-1/c4ca84b7-1551-4c2c-8820-d3c9b73c3358/scratchpad/webapp-${device}.png`, fullPage: true });
  await page.close();
}

await browser.close();
console.log(failed ? `\n${failed} CHECK(S) FAILED` : '\nALL E2E CHECKS PASSED');
process.exit(failed ? 1 : 0);
