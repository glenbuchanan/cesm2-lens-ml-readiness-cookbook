const plugin = {
  name: 'a11y-headings',
  transforms: [
    {
      name: 'a11y-heading-levels',
      doc: 'Re-level headings so none is more than one level deeper than the one before it and siblings share a depth (the lifted title is h1).',
      stage: 'document',
      plugin: (_opts, utils) => (tree) => {
        const headings = utils.selectAll('heading', tree);
        const assigned = new Map(); // source depth -> depth given to the last heading at that source depth (current context)
        let prev = 1;               // the page title is the h1 the theme renders above the body
        for (const h of headings) {
          const d = h.depth;
          if (typeof d !== 'number') continue;
          for (const k of [...assigned.keys()]) if (k > d) assigned.delete(k); // a shallower heading ends its children's context
          const next = assigned.has(d) ? assigned.get(d) : Math.min(d, prev + 1);
          assigned.set(d, next);
          h.depth = next;
          prev = next;
        }
      },
    },
  ],
};
export default plugin;
