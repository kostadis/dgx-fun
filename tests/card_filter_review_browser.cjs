// Run with PLAYWRIGHT_MODULE pointing to an installed Playwright package.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const {pathToFileURL}=require('node:url');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE?{executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1100,height:900},acceptDownloads:true});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(process.argv[2]).href);
  const meta=await page.evaluate(()=>({total:items.length,high:items.filter(it=>tierOf(it)==='high').map(it=>it.id),middle:items.filter(it=>tierOf(it)==='middle').map(it=>it.id),other:items.filter(it=>tierOf(it)==='cards').map(it=>it.id)}));
  assert(meta.high.length>=2&&meta.middle.length>=2&&meta.other.length>=2);
  assert.equal(await page.locator('#save').isDisabled(),true);
  assert.equal(await page.evaluate(()=>Object.keys(decisions).length),0);
  // An exception before bulk approval must remain a card and retain its choice.
  const except=meta.high[0],other=meta.other[0];
  await page.locator(`[data-except="${except}"]`).click();
  await page.locator(`[data-card="${except}"] [data-choice="reject"]`).click();
  await page.locator(`[data-card="${other}"] [data-choice="discuss"]`).click();
  await page.locator(`[data-note="${other}"]`).fill('Check the independent transcript.');
  await page.locator('#bulk-high').click();
  let output=await page.evaluate(()=>payload());
  assert.equal(output.decisions[except],'reject');
  assert.equal(output.decisions[other],'discuss');
  assert.equal(output.notes[other],'Check the independent transcript.');
  assert.equal(output.tally.approve,meta.high.length-1);
  assert(meta.middle.every(id=>!output.decisions[id]));
  assert.equal(output.unmarked.length,meta.other.length+meta.middle.length-1);
  // Pulling a row out after bulk approval revokes that approval.
  const after=meta.high[1];await page.locator(`[data-except="${after}"]`).click();
  output=await page.evaluate(()=>payload());
  assert.equal(output.decisions[after],undefined);
  assert(output.unmarked.includes(after));
  // The middle group's bulk approval must not approve that regular-card exception.
  await page.locator('#bulk-middle').click();
  output=await page.evaluate(()=>payload());
  assert(meta.middle.every(id=>output.decisions[id]==='approve'));
  assert.equal(output.decisions[after],undefined);
  assert.equal(output.decisions[except],'reject');
  assert.equal(output.notes[other],'Check the independent transcript.');
  // Middle rows can also be pulled out after their group's approval.
  const middleAfter=meta.middle[0];await page.locator(`[data-except="${middleAfter}"]`).click();
  output=await page.evaluate(()=>payload());
  assert.equal(output.decisions[middleAfter],undefined);
  assert(output.unmarked.includes(middleAfter));
  const [download]=await Promise.all([page.waitForEvent('download'),page.locator('#save').click()]);
  await download.saveAs(process.argv[3]);
  const saved=JSON.parse(await fs.readFile(process.argv[3],'utf8'));
  assert(saved.savedAt&&saved.replay);
  assert.equal(saved.decided+saved.unmarked.length,meta.total);
  await page.reload();
  assert.equal(await page.evaluate(()=>Object.keys(decisions).length),0);
  assert.equal(await page.locator('#save').isDisabled(),true);
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path:process.argv[4],fullPage:false});
  if(process.argv[5]) await page.locator('#middle-section').screenshot({path:process.argv[5]});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({items:meta.total,over90:meta.high.length,from80to90:meta.middle.length,regularCards:meta.other.length,exported:saved.decided,browserErrors:errors}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
