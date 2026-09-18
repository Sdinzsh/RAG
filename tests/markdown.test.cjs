const { test } = require('node:test');
const assert = require('node:assert/strict');
const { renderMarkdown } = require('../static/markdown.js');

test('document/model HTML and remote media cannot execute or load', () => {
  const html = renderMarkdown('<img src="https://example.com/x" onerror="alert(1)">\n![tracking](https://example.com/x)\n<script>alert(1)</script>');
  assert.ok(!/<(?:img|script)\b/.test(html));
  assert.ok(html.includes('&lt;img'));
});

test('headings, code, lists and tables render offline', () => {
  const html = renderMarkdown('# Revenue\n\n- **Premium**\n- Ads\n\n```html\n<img>\n```\n\nName | Value\n--- | ---\nIncome | 15');
  assert.ok(html.includes('<h1>Revenue</h1>'));
  assert.ok(html.includes('<li><strong>Premium</strong></li>'));
  assert.ok(html.includes('<pre><code>&lt;img&gt;</code></pre>'));
  assert.ok(html.includes('<td>15</td>'));
});

test('unfinished streaming code fence stays escaped', () => {
  assert.ok(renderMarkdown('```\n<script>').includes('&lt;script&gt;'));
});
