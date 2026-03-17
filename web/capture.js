const puppeteer = require('puppeteer');

(async () => {
    // wait a few seconds before generating
    await new Promise(r => setTimeout(r, 6000));
    try {
        const browser = await puppeteer.launch();
        const page = await browser.newPage();
        await page.setViewport({ width: 1440, height: 900 });

        await page.goto('http://localhost:3001', { waitUntil: 'networkidle2' });
        await page.screenshot({ path: 'dark_mode_home.png', fullPage: true });

        // Open sidebar slightly
        await page.click('button[aria-label="打开导航"]');
        await new Promise(r => setTimeout(r, 1000));

        await browser.close();
        console.log("Screenshot generated.");
    } catch (err) {
        console.error(err);
    }
})();
