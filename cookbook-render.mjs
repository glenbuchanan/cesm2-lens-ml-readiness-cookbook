// cookbook-render.mjs — one build-time fix for the rendered book.
//
// figure-alt: a matplotlib output carries text/plain "<Figure size 990x418 with 1 Axes>", and the book theme writes that string
// into the <img alt>. Each figure cell in the notebook carries a descriptive alt in its cell metadata (metadata.alt); this
// transform copies it into the output's text/plain so the rendered <img alt> describes the figure. A figure cell without an
// alt, or any matplotlib repr left in an image output anywhere on the page (say, because mystmd's tree changed and the transform
// no longer finds it), fails the build with an error naming the page: an alt that silently reads "<Figure size …>" is not shipped.

const FIGURE_REPR = /^<Figure size [^>]*>$/;

function pageName(file) {
  const p = (file && (file.path || file.history?.[0])) || '';
  return p.replace(/^.*?(notebooks\/|README)/, '$1') || '(page)';
}

const plugin = {
  name: 'cookbook-render',
  transforms: [
    {
      name: 'figure-alt',
      doc: "Copy each figure cell's metadata.alt into the image output's text/plain, which the theme renders as the <img alt>.",
      stage: 'document',
      plugin: (_opts, utils) => (tree, file) => {
        const page = pageName(file);
        let fixed = 0;
        for (const block of utils.selectAll('block', tree)) {
          if (block.kind !== 'notebook-code') continue;
          const alt = block.data && typeof block.data.alt === 'string' ? block.data.alt.trim() : '';
          for (const out of utils.selectAll('output', block)) {
            const jd = out.jupyter_data;
            const data = jd && jd.data;
            if (!data || !Object.keys(data).some((k) => k.startsWith('image/'))) continue;
            const plain = Array.isArray(data['text/plain']) ? data['text/plain'].join('') : data['text/plain'];
            if (typeof plain !== 'string' || !FIGURE_REPR.test(plain.trim())) continue;
            if (!alt) {
              console.log(`figure-alt: ${page}: image output "${plain.trim()}" on cell ${block.identifier || '?'} has no metadata.alt`);
              continue; // counted and thrown below
            }
            data['text/plain'] = alt;
            fixed += 1;
          }
        }
        if (fixed) console.log(`figure-alt: ${page}: ${fixed} figure alt(s) written from cell metadata`);
        // The check that can fail: after the transform, no image output anywhere in the page may still carry the matplotlib repr.
        const left = [];
        for (const out of utils.selectAll('output', tree)) {
          const data = out.jupyter_data && out.jupyter_data.data;
          if (!data || !Object.keys(data).some((k) => k.startsWith('image/'))) continue;
          const plain = Array.isArray(data['text/plain']) ? data['text/plain'].join('') : data['text/plain'];
          if (typeof plain === 'string' && FIGURE_REPR.test(plain.trim())) left.push(plain.trim());
        }
        if (left.length) {
          throw new Error(
            `figure-alt: ${page}: ${left.length} image output(s) still carry the matplotlib repr as alt text ` +
              `(${left.join('; ')}) after ${fixed} fix(es). Add metadata.alt to every figure cell, or update cookbook-render.mjs ` +
              `if mystmd's output tree changed.`,
          );
        }
      },
    },
  ],
};
export default plugin;
