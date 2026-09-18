// Small offline Markdown renderer. Raw HTML and remote media remain plain text.
// Supports headings, lists, quotes, fenced code, tables, bold and inline code.
(function (root) {
  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  function inline(text) {
    return text.split(/(`[^`]+`)/g).map(part => {
      if (part.startsWith('`') && part.endsWith('`')) {
        return '<code>' + escapeHtml(part.slice(1, -1)) + '</code>';
      }
      return escapeHtml(part).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    }).join('');
  }

  function renderMarkdown(text) {
    const lines = String(text).replace(/\r\n/g, '\n').split('\n');
    const output = [];
    let paragraph = [], list = null, code = null, fence = null;
    function flush() {
      if (paragraph.length) output.push('<p>' + paragraph.map(inline).join('<br>') + '</p>');
      paragraph = [];
      if (list) { output.push('</' + list + '>'); list = null; }
    }
    function cells(line) {
      return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(x => x.trim());
    }
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const marker = line.match(/^\s{0,3}(`{3,}|~{3,})(.*)$/);
      if (code !== null) {
        if (marker && marker[1][0] === fence[0] && marker[1].length >= fence.length && !marker[2].trim()) {
          output.push('<pre><code>' + escapeHtml(code.join('\n')) + '</code></pre>');
          code = null;
        } else code.push(line);
        continue;
      }
      if (marker) { flush(); code = []; fence = marker[1]; continue; }
      if (!line.trim()) { flush(); continue; }
      const heading = line.match(/^(#{1,6})\s+(.+)$/);
      if (heading) {
        flush(); const h = heading[1].length;
        output.push('<h' + h + '>' + inline(heading[2]) + '</h' + h + '>'); continue;
      }
      if (line.includes('|') && i + 1 < lines.length &&
          cells(lines[i + 1]).every(cell => /^:?-{3,}:?$/.test(cell))) {
        flush();
        output.push('<table><thead><tr>' + cells(line).map(cell => '<th>' + inline(cell) + '</th>').join('') + '</tr></thead><tbody>');
        i++;
        while (i + 1 < lines.length && lines[i + 1].includes('|') && lines[i + 1].trim()) {
          output.push('<tr>' + cells(lines[++i]).map(cell => '<td>' + inline(cell) + '</td>').join('') + '</tr>');
        }
        output.push('</tbody></table>'); continue;
      }
      const item = line.match(/^\s*(?:([-*+])|\d+\.)\s+(.+)$/);
      if (item) {
        const type = item[1] ? 'ul' : 'ol';
        if (list !== type) { flush(); list = type; output.push('<' + list + '>'); }
        output.push('<li>' + inline(item[2]) + '</li>'); continue;
      }
      if (list) flush();
      if (line.startsWith('> ')) { flush(); output.push('<blockquote>' + inline(line.slice(2)) + '</blockquote>'); continue; }
      paragraph.push(line);
    }
    flush();
    if (code !== null) output.push('<pre><code>' + escapeHtml(code.join('\n')) + '</code></pre>');
    return output.join('\n');
  }
  if (typeof module !== 'undefined') module.exports = { renderMarkdown };
  else root.renderMarkdown = renderMarkdown;
})(typeof window === 'undefined' ? globalThis : window);
